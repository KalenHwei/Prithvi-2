import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# ----------------------------
# Core blocks
# ----------------------------

class PatchEmbed2D(nn.Module):
    """
    2D patch embedding for images (applied per time step if T>1).
    Input shape: (B, T, C, H, W) or (B, C, H, W)
    Output: (B, T, D, H', W') and (H', W')
    """
    def __init__(self, in_chans: int, embed_dim: int = 256, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.dim() == 4:
            # (B, C, H, W) -> add T=1
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)  # (B*T, D, H', W')
        Hp, Wp = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D), N=Hp*Wp
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, T, -1, Hp, Wp)  # (B, T, D, H', W')
        return x, (Hp, Wp)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalBlock(nn.Module):
    """
    iTransformer-style temporal mixer: self-attention along the time axis
    for each spatial location (token) independently.
    Input: (B, T, D, H, W)
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D, H, W = x.shape
        # arrange tokens as sequences over time for each spatial location
        x_ = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)  # (B*H*W, T, D)
        x_norm = self.norm1(x_)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_ = x_ + attn_out
        x_ = x_ + self.mlp(self.norm2(x_))
        x = x_.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)  # (B, T, D, H, W)
        return x


class SpatialConvBlock(nn.Module):
    """Lightweight spatial mixer with depthwise separable convs (per time step)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D, H, W)
        B, T, D, H, W = x.shape
        x_ = x.reshape(B * T, D, H, W)
        y = self.dw(x_)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        y = y.reshape(B, T, D, H, W)
        return y


class Downsample(nn.Module):
    """2x spatial downsampling, keep channels (or increase via out_dim)."""
    def __init__(self, in_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D, H, W)
        B, T, D, H, W = x.shape
        x_ = x.reshape(B * T, D, H, W)
        x_ = self.conv(x_)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        Ho, Wo = x_.shape[-2:]
        x_ = x_.reshape(B, T, -1, Ho, Wo)
        return x_


# ----------------------------
# iTransformer backbone
# ----------------------------

class ITransformerBackbone(nn.Module):

    def __init__(
        self,
        in_chans: int,
        embed_dim: int = 256,
        depths: Tuple[int, int, int, int] = (2, 2, 4, 2),
        num_heads: Tuple[int, int, int, int] = (4, 4, 8, 8),
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2D(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)

        dims = [embed_dim, embed_dim, embed_dim * 2, embed_dim * 2]

        # Stage 1 (stride 4)
        self.stage1 = nn.ModuleList([
            nn.Sequential(TemporalBlock(dims[0], num_heads[0], mlp_ratio, drop_rate, attn_drop_rate),
                          SpatialConvBlock(dims[0]))
            for _ in range(depths[0])
        ])
        self.ds1 = Downsample(dims[0], dims[1])  # -> stride 8

        # Stage 2 (stride 8)
        self.stage2 = nn.ModuleList([
            nn.Sequential(TemporalBlock(dims[1], num_heads[1], mlp_ratio, drop_rate, attn_drop_rate),
                          SpatialConvBlock(dims[1]))
            for _ in range(depths[1])
        ])
        self.ds2 = Downsample(dims[1], dims[2])  # -> stride 16

        # Stage 3 (stride 16)
        self.stage3 = nn.ModuleList([
            nn.Sequential(TemporalBlock(dims[2], num_heads[2], mlp_ratio, drop_rate, attn_drop_rate),
                          SpatialConvBlock(dims[2]))
            for _ in range(depths[2])
        ])
        self.ds3 = Downsample(dims[2], dims[3])  # -> stride 32

        # Stage 4 (stride 32)
        self.stage4 = nn.ModuleList([
            nn.Sequential(TemporalBlock(dims[3], num_heads[3], mlp_ratio, drop_rate, attn_drop_rate),
                          SpatialConvBlock(dims[3]))
            for _ in range(depths[3])
        ])

        self.out_dims = (dims[0], dims[1], dims[2], dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, C, H, W) or (B, T, C, H, W)
        x, _ = self.patch_embed(x)  # (B, T, D, H/4, W/4)

        # Stage 1
        for blk in self.stage1:
            x = blk(x)
        f1 = x.mean(dim=1)  # (B, D1, H/4, W/4)

        # Stage 2
        x = self.ds1(x)
        for blk in self.stage2:
            x = blk(x)
        f2 = x.mean(dim=1)  # (B, D2, H/8, W/8)

        # Stage 3
        x = self.ds2(x)
        for blk in self.stage3:
            x = blk(x)
        f3 = x.mean(dim=1)  # (B, D3, H/16, W/16)

        # Stage 4
        x = self.ds3(x)
        for blk in self.stage4:
            x = blk(x)
        f4 = x.mean(dim=1)  # (B, D4, H/32, W/32)

        return [f1, f2, f3, f4]


class FPNDecoder(nn.Module):
    def __init__(self, in_dims: Tuple[int, int, int, int], out_dim: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_dim, kernel_size=1) for c in in_dims])
        self.fuse_convs = nn.ModuleList([nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1) for _ in in_dims])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: [C2, C3, C4, C5] (low to high stride)
        c2, c3, c4, c5 = feats
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        p5 = self.fuse_convs[3](p5)
        p4 = self.fuse_convs[2](p4)
        p3 = self.fuse_convs[1](p3)
        p2 = self.fuse_convs[0](p2)

        # aggregate at p2 resolution
        out = p2
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, head_channels: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        c1, c2 = head_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(c2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class ITransformerSegmentationModel(nn.Module):
    def __init__(
        self,
        in_chans: int,
        num_classes: int = 2,
        embed_dim: int = 256,
        depths: Tuple[int, int, int, int] = (2, 2, 4, 2),
        num_heads: Tuple[int, int, int, int] = (4, 4, 8, 8),
        head_channels: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        patch_size: int = 4,
    ):
        super().__init__()
        self.backbone = ITransformerBackbone(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            patch_size=patch_size,
        )
        self.decoder = FPNDecoder(self.backbone.out_dims, out_dim=256)
        self.head = SegmentationHead(256, num_classes=num_classes, head_channels=head_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, T, C, H, W)
        feats = self.backbone(x)
        p2 = self.decoder(feats)
        # upsample logits to input image size
        if x.dim() == 5:
            H, W = x.shape[-2:]
            B = x.shape[0]
        else:
            B, _, H, W = x.shape
        logits = self.head(p2)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (B, C, H, W), target: (B, H, W)
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            # optional class balance (binary: weight class 1)
            with torch.no_grad():
                w = torch.ones_like(target, dtype=loss.dtype, device=loss.device)
                w = w * (self.alpha)
                loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Mean IoU (Jaccard) over classes (ignore index -1)."""
    # pred: (B, H, W) class ids, target: (B, H, W)
    mask = target != -1
    pred = pred[mask]
    target = target[mask]
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        tgt_c = target == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union == 0:
            iou = torch.tensor(1.0, device=pred.device)  # ignore class absent in both
        else:
            iou = inter / union
        ious.append(iou)
    return torch.stack(ious).mean()


class ITransformerSegTask(pl.LightningModule):
    def __init__(
        self,
        in_chans: int,
        num_classes: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        loss: str = "focal",
        head_channels: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        scheduler_step: int = 10,
        scheduler_gamma: float = 0.9,
        **backbone_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ITransformerSegmentationModel(
            in_chans=in_chans,
            num_classes=num_classes,
            head_channels=head_channels,
            dropout=dropout,
            **backbone_kwargs,
        )
        if loss == "focal":
            self.criterion = FocalLoss()
        elif loss == "ce":
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise ValueError(f"Unknown loss: {loss}")
        self.num_classes = num_classes
        self.lr = lr
        self.wd = weight_decay
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        if isinstance(batch, dict):
            x = batch.get("image")
            y = batch.get("mask")
        else:
            x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            miou = compute_iou(preds, y, self.num_classes)
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), sync_dist=True)
        self.log(f"{stage}/Multiclass_Jaccard_Index", miou, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}