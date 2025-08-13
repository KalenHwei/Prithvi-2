# itransformer.py
# 将 Prithvi (ViT) 骨干作为“空间编码器”完整复用（可加载你现有的 .pt 权重），
# 在其输出的中间层特征上，叠加一个“iTransformer 风格”的时序模块（仅沿时间维做注意力与前馈），
# 最后接一个轻量 FPN/UPerNet 风格解码器做语义分割。
#
# 兼容输入：
#   - 静态: (B, C, H, W)
#   - 时序: (B, T, C, H, W)
#
# 注意：
# 1) 该实现不修改 Prithvi 的空间结构，因而能最大化复用 checkpoint；
#    新增的 TemporalMixer 参数随机初始化，默认可训练。
# 2) “抓中间层特征”使用通用的“Block hooks”方式，不依赖具体类名；
#    只要 Prithvi 的 forward 内部按顺序调用 Transformer blocks（ModuleList/Sequential），就能截取。
# 3) Patch 大小需与你的 Prithvi 匹配（默认 16），请按实际模型设置。
# 4) 如果你的 checkpoint 是 state_dict 而不是整模型，本脚本提供了非严格匹配加载（strict=False），
#    但你需要用相同结构实例化骨干；这里优先尝试直接 load 一个保存的 nn.Module。
#
# 作者：你现在的 ChatGPT 助手

from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# 小工具：Focal Loss（和你原脚本一致的名字）
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B, C, H, W), targets: (B, H, W)
        num_classes = logits.shape[1]
        ce = F.cross_entropy(
            logits, targets.long(),
            reduction="none",
            ignore_index=self.ignore_index
        )  # (B,H,W)
        with torch.no_grad():
            pt = torch.exp(-ce)  # = softmax prob of the true class
        focal = (self.alpha * (1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            valid = (targets != self.ignore_index).float()
            return (focal * valid).sum() / (valid.sum().clamp_min(1.0))
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# --------------------------
# 通用：寻找 transformer blocks 并挂钩
# --------------------------
def find_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    """
    尽可能通用地搜集“Transformer Block”列表：
    - 优先找 model.blocks (ModuleList)
    - 其次尝试抓取包含子模块且具有 self-attn/ffn 结构的模块
    """
    # 1) 直接找属性名为 'blocks' 的模块列表
    if hasattr(model, "blocks") and isinstance(model.blocks, (nn.ModuleList, nn.Sequential)):
        return list(model.blocks)

    # 2) 否则遍历，收集疑似 transformer block（含 attn/attention + mlp/feed_forward）
    candidates = []
    for m in model.modules():
        has_attn = any(hasattr(m, name) for name in ["attn", "attention", "self_attn", "self_attention"])
        has_ffn  = any(hasattr(m, name) for name in ["mlp", "ffn", "feed_forward"])
        if has_attn and has_ffn:
            # 排除顶层模型本身
            if m is not model:
                candidates.append(m)
    # 去重且按发现顺序保留
    uniq = []
    seen = set()
    for m in candidates:
        if id(m) not in seen:
            uniq.append(m)
            seen.add(id(m))
    return uniq


class BlockFeatureHook:
    """
    在给定的 block 索引上注册 forward hook，记录每个 block 的输出张量。
    """
    def __init__(self, blocks: List[nn.Module], indices: List[int]):
        self.blocks = blocks
        self.indices = sorted(indices)
        self.handles = []
        self.cache: Dict[int, torch.Tensor] = {}

    def __enter__(self):
        def _make_hook(idx):
            def hook(module, inp, out):
                # out 预期为 (B, N_tokens(+cls?), D)
                self.cache[idx] = out
            return hook

        for i, b in enumerate(self.blocks):
            if i in self.indices:
                h = b.register_forward_hook(_make_hook(i))
                self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --------------------------
# iTransformer 风格：仅沿“时间维 T”做注意力和前馈（不搅动空间 token）
# --------------------------
class TemporalBlock(nn.Module):
    """
    输入形状： (B, T, N, D)
    对每个 token 位置（N个）做时间注意力：把 (B,N) 两维合并做 MHA，再还原。
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D)
        B, T, N, D = x.shape
        x_ = x.view(B * N, T, D)
        # MHA
        y = self.attn(self.norm1(x_), self.norm1(x_), self.norm1(x_), need_weights=False)[0]
        x_ = x_ + y
        # MLP
        y2 = self.mlp(self.norm2(x_))
        x_ = x_ + y2
        return x_.view(B, T, N, D)


class TemporalMixer(nn.Module):
    """
    堆叠 L 个 TemporalBlock；支持 mean/last/attnpool 时间聚合。
    """
    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, pool: str = "mean"):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.pool = pool
        if pool == "attnpool":
            self.q = nn.Parameter(torch.randn(1, 1, dim))  # 可学习 query，做跨时间注意力池化
            self.norm = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D)
        for blk in self.blocks:
            x = blk(x)
        if x.shape[1] == 1:
            return x[:, 0]  # (B,N,D)
        if self.pool == "mean":
            return x.mean(dim=1)  # (B,N,D)
        if self.pool == "last":
            return x[:, -1]      # (B,N,D)
        if self.pool == "attnpool":
            B, T, N, D = x.shape
            q = self.q.expand(B * N, 1, D)
            kv = self.norm(x.view(B * N, T, D))
            out = self.attn(q, kv, kv, need_weights=False)[0]  # (B*N,1,D)
            return out.view(B, N, D)
        raise ValueError(f"Unknown pool: {self.pool}")


# --------------------------
# 轻量 FPN/UPerNet 解码器
# --------------------------
class FPNDecoder(nn.Module):
    """
    输入：来自 4 个层级的特征 [P2, P3, P4, P5]，每个为 (B, N_i, D)
    我们先把 token 还原为 2D (B, D, H_i, W_i)，做自顶向下融合并输出最终分割。
    """
    def __init__(self, dims: List[int], out_channels: int = 256, head_channels: Tuple[int, int] = (128, 64),
                 num_classes: int = 2, patch_size: int = 16, in_hw: Tuple[int, int] = (224, 224), dropout: float = 0.1):
        super().__init__()
        self.patch = patch_size
        self.in_hw = in_hw
        self.lateral = nn.ModuleList([nn.Conv2d(d, out_channels, 1) for d in dims])
        self.smooth  = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in dims[:-1]])
        c1, c2 = head_channels
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c2, num_classes, 1)
        )

    def tokens_to_map(self, x_tokens: torch.Tensor, dim: int) -> torch.Tensor:
        # x_tokens: (B, N, dim) -> (B, dim, H, W)
        B, N, D = x_tokens.shape
        H = self.in_hw[0] // self.patch
        W = self.in_hw[1] // self.patch
        assert H * W == N, f"Token count {N} mismatch with HxW {H}x{W}"
        return x_tokens.transpose(1, 2).reshape(B, D, H, W)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: [P2, P3, P4, P5]，每个 (B, N, D_i) —— 已做完时间聚合
        """
        assert len(feats) == 4, "Expect 4 pyramid levels"
        maps = []
        for i, f in enumerate(feats):
            maps.append(self.lateral[i](self.tokens_to_map(f, f.shape[-1])))

        # top-down: maps[-1] 是最高层，逐级上采样 + smooth
        p5, p4, p3, p2 = maps[3], maps[2], maps[1], maps[0]
        p4 = self.smooth[2](p4 + F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False))
        p3 = self.smooth[1](p3 + F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth[0](p2 + F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False))

        out = self.head(F.interpolate(p2, scale_factor=self.patch, mode="bilinear", align_corners=False))
        return out  # (B, num_classes, H, W) 与输入分辨率对齐


# --------------------------
# 主模型：Prithvi 空间编码 + 时序混合 + FPN 解码
# --------------------------
class IPrithviSTModel(nn.Module):
    def __init__(
        self,
        prithvi_module: nn.Module,
        feature_indices: List[int] = [7, 15, 23, 31],
        temporal_depth: int = 2,
        temporal_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,
        temporal_dropout: float = 0.0,
        temporal_pool: str = "mean",
        dims_per_level: List[int] = [768, 768, 768, 768],  # 需与你的 Prithvi 隐藏维度匹配
        num_classes: int = 2,
        in_hw: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        head_channels: Tuple[int, int] = (128, 64),
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = prithvi_module
        self.patch = patch_size
        self.in_hw = in_hw
        self.feature_indices = feature_indices

        # temporal mixer for each level
        self.temporal_mixers = nn.ModuleList([
            TemporalMixer(dim=dims_per_level[i],
                          depth=temporal_depth,
                          num_heads=temporal_heads,
                          mlp_ratio=temporal_mlp_ratio,
                          dropout=temporal_dropout,
                          pool=temporal_pool)
            for i in range(4)
        ])

        self.decoder = FPNDecoder(
            dims=dims_per_level,
            out_channels=256,
            head_channels=head_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            in_hw=in_hw,
            dropout=head_dropout
        )

    @torch.no_grad()
    def _infer_token_dim(self) -> int:
        # 粗略从 backbone 的某个线性层/attn 投影里猜测隐藏维度；若失败则退回 768
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        return 768

    def _run_one_frame(self, x1: torch.Tensor, blocks: List[nn.Module]) -> Dict[int, torch.Tensor]:
        """
        对单帧输入 x1: (B,C,H,W) 运行 backbone，一边运行一边用 hook 抓取 feature_indices 指定的 block 输出。
        返回：{block_idx: (B,N,D)}
        """
        cache: Dict[int, torch.Tensor] = {}
        with BlockFeatureHook(blocks, self.feature_indices) as h:
            _ = self.backbone(x1)  # 无需用返回值，hooks 会把中间输出写进 h.cache
            cache = {k: v for k, v in h.cache.items()}
        # 剥掉 CLS（如果有的话）；有些 ViT 输出 (B, 1+N, D)
        for k, v in cache.items():
            if v.dim() == 3 and v.size(1) > 0:
                # 如果 token 数 == 1 + patch_num，去掉第一个
                H_p = self.in_hw[0] // self.patch
                W_p = self.in_hw[1] // self.patch
                N = H_p * W_p
                if v.size(1) == N + 1:
                    cache[k] = v[:, 1:, :]  # (B,N,D)
        return cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) 或 (B,T,C,H,W)
        输出: (B,num_classes,H,W)
        """
        blocks = find_transformer_blocks(self.backbone)
        assert len(blocks) > max(self.feature_indices), \
            f"Backbone blocks ({len(blocks)}) < max feature index {max(self.feature_indices)}"

        is_seq = (x.dim() == 5)
        if not is_seq:
            x = x.unsqueeze(1)  # (B,1,C,H,W)

        B, T, C, H, W = x.shape
        feat_per_level: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.feature_indices}

        for t in range(T):
            cache = self._run_one_frame(x[:, t], blocks)  # {block_idx: (B,N,D)}
            for idx in self.feature_indices:
                feat_per_level[idx].append(cache[idx])  # list of (B,N,D)

        # 组装为 4 个层级（按 indices 排序）
        feats_T = []
        for i, idx in enumerate(sorted(self.feature_indices)):
            f_list = feat_per_level[idx]                     # T * (B,N,D)
            f = torch.stack(f_list, dim=1)                  # (B,T,N,D)
            f = self.temporal_mixers[i](f)                  # (B,N,D) —— 已按时间聚合
            feats_T.append(f)

        # 解码
        out = self.decoder(feats_T)  # (B,num_classes,H,W)
        return out


# --------------------------
# Lightning Task（与 run 脚本配合）
# --------------------------
import lightning.pytorch as pl
from torchmetrics import JaccardIndex

class IPrithviSegTask(pl.LightningModule):
    def __init__(
        self,
        prithvi_module: nn.Module,
        num_classes: int = 2,
        in_hw: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        feature_indices: List[int] = [7, 15, 23, 31],
        dims_per_level: List[int] = [768, 768, 768, 768],
        temporal_depth: int = 2,
        temporal_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,
        temporal_dropout: float = 0.0,
        temporal_pool: str = "mean",
        head_channels: Tuple[int, int] = (128, 64),
        head_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        loss: str = "focal",
        ignore_index: int = -1,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        unfreeze_backbone_at: int = -1,
        unfreeze_decoder_at: int = -1,
        bn_eval_when_frozen: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["prithvi_module"])

        self.model = IPrithviSTModel(
            prithvi_module=prithvi_module,
            feature_indices=feature_indices,
            temporal_depth=temporal_depth,
            temporal_heads=temporal_heads,
            temporal_mlp_ratio=temporal_mlp_ratio,
            temporal_dropout=temporal_dropout,
            temporal_pool=temporal_pool,
            dims_per_level=dims_per_level,
            num_classes=num_classes,
            in_hw=in_hw,
            patch_size=patch_size,
            head_channels=head_channels,
            head_dropout=head_dropout,
        )

        self.criterion = FocalLoss(ignore_index=ignore_index) if loss == "focal" else nn.CrossEntropyLoss(ignore_index=ignore_index)

        task_type = "binary" if num_classes == 2 else "multiclass"
        self.val_iou = JaccardIndex(task=task_type, num_classes=num_classes, ignore_index=ignore_index)
        self.test_iou = JaccardIndex(task=task_type, num_classes=num_classes, ignore_index=ignore_index)

        # 冻结/解冻
        if freeze_backbone:
            self._set_trainable(self.model.backbone, False, set_eval=bn_eval_when_frozen)
        if freeze_decoder:
            for m in [self.model.decoder]:
                self._set_trainable(m, False, set_eval=bn_eval_when_frozen)

        self.unfreeze_backbone_at = unfreeze_backbone_at
        self.unfreeze_decoder_at = unfreeze_decoder_at
        self.bn_eval_when_frozen = bn_eval_when_frozen

        self.lr = lr
        self.wd = weight_decay
        self.ignore_index = ignore_index

    # ====== 冻结/解冻工具 ======
    def _set_trainable(self, module: nn.Module, trainable: bool, set_eval: bool = False):
        for p in module.parameters():
            p.requires_grad = trainable
        if set_eval and not trainable:
            module.eval()

    def _ensure_optimizer_has(self, module: nn.Module):
        opt = self.optimizers(use_pl_optimizer=False)
        if opt is None:
            return
        existing = {id(p) for g in opt.param_groups for p in g["params"]}
        new_params = [p for p in module.parameters() if p.requires_grad and id(p) not in existing]
        if new_params:
            opt.add_param_group({"params": new_params, "lr": self.lr, "weight_decay": self.wd})

    def on_train_epoch_start(self):
        if self.unfreeze_backbone_at >= 0 and self.current_epoch == self.unfreeze_backbone_at:
            self._set_trainable(self.model.backbone, True, set_eval=False)
            self.model.backbone.train()
            self._ensure_optimizer_has(self.model.backbone)

        if self.unfreeze_decoder_at >= 0 and self.current_epoch == self.unfreeze_decoder_at:
            self._set_trainable(self.model.decoder, True, set_eval=False)
            self.model.decoder.train()
            self._ensure_optimizer_has(self.model.decoder)

    # ====== 训练/验证/测试 ======
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def _shared_step(self, batch, stage: str):
        x, y = batch["image"], batch["mask"]  # 兼容 Landslide4SenseNonGeoDataModule
        logits = self(x)
        loss = self.criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        if stage == "val":
            iou = self.val_iou(pred, y)
            self.log("val/Multiclass_Jaccard_Index", iou, prog_bar=True, sync_dist=True)
        elif stage == "test":
            iou = self.test_iou(pred, y)
            self.log("test/Multiclass_Jaccard_Index", iou, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "test")
        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss


# --------------------------
# 加载 Prithvi 模块的辅助函数
# --------------------------
def load_prithvi_module(ckpt_path: str) -> nn.Module:
    """
    优先尝试直接 load 一个保存的 nn.Module；
    若是 state_dict，尝试在 strict=False 下 load 到同结构实例（这里无法构造具体骨干，抛出清晰错误）。
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, nn.Module):
        return obj
    elif isinstance(obj, dict):
        # 可能是 {"state_dict":..., "model":..., "config":...}
        sd = obj.get("model", None)
        if sd is None:
            sd = obj.get("state_dict", None)
        if isinstance(sd, dict):
            raise RuntimeError(
                "检测到是 state_dict，但缺少具体骨干构造代码。\n"
                "请在你的环境里用 terratorch 的 Prithvi 构造同结构模型，然后用 strict=False 加载权重，"
                "再把该 nn.Module 传入 IPrithviSegTask(prithvi_module=...)."
            )
    raise RuntimeError(f"不认识的 checkpoint 格式：{type(obj)}")
