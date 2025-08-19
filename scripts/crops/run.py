import os
import numpy as np
import torch
import argparse
import json

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

import albumentations as A
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/dataset_for_temporal_corp/data", help="Path to the dataset.")
    parser.add_argument("--log_saving_path", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--backbone_path", type=str, default="checkpoints", help="Path to pretrained backbone checkpoints.")
    parser.add_argument("--checkpoint_saving_path", type=str, default="finetuned_checkpoints", help="Directory to save finetuned model checkpoints.")
    parser.add_argument("--backbone", type=str, default="prithvi_eo_v2_600_tl")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--bands", dest="BANDS", default=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"], help="List of bands to use.")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of temporal frames.")
    parser.add_argument("--class_weights", type=list, default=[0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462, 1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702], help="Number of classes for classification.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model head.")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for the optimizer.")

    parser.add_argument("--freeze_backbone", type=bool, default=True, help="Freeze the backbone during training.")
    
    parser.add_argument("--plot_on_val", action="store_true", help="Plot validation samples during training.")
    parser.add_argument("--device", type=list, default=[0,1,2,3], help="List of GPU devices to use.")

    return parser.parse_args()

def main():
    cfg = get_args()

    if cfg.backbone == "prithvi_eo_v2_600_tl":
        backbone_name = "Prithvi_EO_V2_600M_TL.pt"
    elif cfg.backbone == "prithvi_eo_v2_600":
        backbone_name = "Prithvi_EO_V2_600.pt"
    elif cfg.backbone == "prithvi_eo_v2_300_tl":
        backbone_name = "Prithvi_EO_V2_300_TL.pt"
    elif cfg.backbone == "prithvi_eo_v2_300":
        backbone_name = "Prithvi_EO_V2_300.pt"
    else:
        raise ValueError(f"backbone {cfg.backbone} not supported")


    train_transforms = [
        terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
        A.Flip(),
        A.pytorch.transforms.ToTensorV2(),
        terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=cfg.num_frames),
    ]

    log_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="multi-temporal-crop"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/mIoU",
        mode="max",
        dirpath=checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1
    )

    pl.seed_everything(cfg.seed)

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy="auto",
        devices=cfg.device,
        precision="bf16-mixed",
        num_nodes=1,
        logger=logger,
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        limit_predict_batches=1,  # predict only in the first batch for generating plots
    )

    data_module = MultiTemporalCropClassificationDataModule(
        batch_size=cfg.batch_size,
        data_root=cfg.data_path,
        train_transform=train_transforms,
        reduce_zero_label=True,
        expand_temporal_dimension=True,
        num_workers=7,
        use_metadata=True,)
    

    backbone_args = dict(
        backbone=cfg.backbone, 
        backbone_pretrained=False, 
        backbone_ckpt_path=os.path.join(cfg.backbone_path, cfg.backbone, backbone_name),
        backbone_coords_encoding=["time", "location"],
        backbone_bands=cfg.BANDS,
        backbone_num_frames=cfg.num_frames,
    )

    decoder_args = dict(
        decoder="UperNetDecoder",
        decoder_channels=256,
        decoder_scale_modules=True,
    )

    necks = [
        dict(
            name="SelectIndices",
            indices=[7, 15, 23, 31],  # indices=[5, 11, 17, 23] for prithvi_eo_v2_300, indices=[7, 15, 23, 31] for prithvi_eo_v2_600
        ),
        dict(name="ReshapeTokensToImage", effective_time_dim=cfg.num_frames),
    ]

    model_args = dict(
        **backbone_args,
        **decoder_args,
        num_classes=len(cfg.class_weights), # 13 classes for this dataset
        head_dropout=cfg.dropout,
        necks=necks,
        rescale=True,
    )

    model = SemanticSegmentationTask(
        model_args=model_args,
        plot_on_val=False,
        class_weights=cfg.class_weights,
        loss="ce",
        lr=cfg.lr,
        optimizer="AdamW",
        optimizer_hparams=dict(weight_decay=cfg.weight_decay),
        ignore_index=-1,
        freeze_backbone=cfg.freeze_backbone,
        freeze_decoder=False,
        model_factory="EncoderDecoderFactory",
    )

    trainer.fit(model, datamodule=data_module)

    ckpt_path = checkpoint_callback.best_model_path
    test_results = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    if test_results:
        results_path = os.path.join(log_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results[0], f, indent=4)
        print(f"Test results: {test_results}")
        print(f"Test results saved to: {results_path}")

if __name__ == "__main__":
    main()