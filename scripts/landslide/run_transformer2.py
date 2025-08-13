
import os
import json
import argparse

import torch
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from terratorch.datamodules import Landslide4SenseNonGeoDataModule

from itransformer import (
    IPrithviSegTask,
    load_prithvi_module,
)

def get_args():
    parser = argparse.ArgumentParser()
    # 数据与日志
    parser.add_argument("--data_path", type=str, default="dataset_for_landslide/data")
    parser.add_argument("--log_saving_path", type=str, default="logs")
    parser.add_argument("--checkpoint_saving_path", type=str, default="finetuned_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=list, default=[0,1,2,3])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)

    # Prithvi checkpoint
    parser.add_argument("--backbone_ckpt", type=str, required=True, help="Prithvi .pt 路径（建议是完整 nn.Module）")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--feature_indices", type=int, nargs="+", default=[7, 15, 23, 31])
    parser.add_argument("--dims_per_level", type=int, nargs="+", default=[768, 768, 768, 768],
                        help="各层 token 维度；需与你的 Prithvi 隐藏维度一致")

    # 训练超参
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bands", dest="BANDS",
                        default=['COASTAL AEROSOL','BLUE','GREEN','RED','RED_EDGE_1','RED_EDGE_2','RED_EDGE_3','NIR_BROAD','WATER_VAPOR','CIRRUS','SWIR_1','SWIR_2','SLOPE','DEM'])
    parser.add_argument("--num_classes", type=int, default=2)

    # 时序模块（iTransformer 风格）
    parser.add_argument("--temporal_depth", type=int, default=2)
    parser.add_argument("--temporal_heads", type=int, default=8)
    parser.add_argument("--temporal_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--temporal_dropout", type=float, default=0.0)
    parser.add_argument("--temporal_pool", type=str, default="mean", choices=["mean","last","attnpool"])

    # 冻结/解冻
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--unfreeze_backbone_at", type=int, default=-1)
    parser.add_argument("--unfreeze_decoder_at", type=int, default=-1)
    parser.add_argument("--bn_eval_when_frozen", action="store_true")

    # 其他
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--loss", type=str, default="focal", choices=["focal","ce"])

    return parser.parse_args()


def main():
    cfg = get_args()

    log_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=log_dir, name="landslide_itransformer")

    checkpoint_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        dirpath=checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1
    )

    # Albumentations
    val_test_tfms = [
        A.Resize(cfg.img_size[0], cfg.img_size[1]),
        A.pytorch.transforms.ToTensorV2(),
    ]
    train_tfms = [
        A.HorizontalFlip(p=0.5),
        A.Resize(cfg.img_size[0], cfg.img_size[1]),
        A.pytorch.transforms.ToTensorV2(),
    ]

    data_module = Landslide4SenseNonGeoDataModule(
        batch_size=cfg.batch_size,
        bands=cfg.BANDS,
        data_root=cfg.data_path,
        train_transform=train_tfms,
        val_transforms=val_test_tfms,
        test_transforms=val_test_tfms,
        num_workers=cfg.num_workers,
    )

    prithvi_module = load_prithvi_module(cfg.backbone_ckpt)

    # Lightning 任务
    model = IPrithviSegTask(
        prithvi_module=prithvi_module,
        num_classes=cfg.num_classes,
        in_hw=(cfg.img_size[0], cfg.img_size[1]),
        patch_size=cfg.patch_size,
        feature_indices=cfg.feature_indices,
        dims_per_level=cfg.dims_per_level,
        temporal_depth=cfg.temporal_depth,
        temporal_heads=cfg.temporal_heads,
        temporal_mlp_ratio=cfg.temporal_mlp_ratio,
        temporal_dropout=cfg.temporal_dropout,
        temporal_pool=cfg.temporal_pool,
        head_channels=(128, 64),
        head_dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        loss=cfg.loss,
        ignore_index=cfg.ignore_index,
        freeze_backbone=cfg.freeze_backbone,
        freeze_decoder=cfg.freeze_decoder,
        unfreeze_backbone_at=cfg.unfreeze_backbone_at,
        unfreeze_decoder_at=cfg.unfreeze_decoder_at,
        bn_eval_when_frozen=cfg.bn_eval_when_frozen,
    )

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy="ddp",
        devices=cfg.device,
        precision="bf16-mixed",
        num_nodes=1,
        logger=logger,
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)
    ckpt_path = checkpoint_callback.best_model_path
    test_results = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    if test_results:
        results_path = os.path.join(log_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results[0], f, indent=4)
        print(f"test results is: {test_results}")
        print(f"Test results saved to: {results_path}")


if __name__ == "__main__":
    main()
