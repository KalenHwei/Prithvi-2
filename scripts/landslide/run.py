import os
import numpy as np
import torch

import terratorch
from terratorch.datamodules import Landslide4SenseNonGeoDataModule
from terratorch.tasks import SemanticSegmentationTask

import albumentations as A

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset_for_landslide/data")
    parser.add_argument("--log_saving_path", type=str, default="logs")
    parser.add_argument("--backbone_path", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_saving_path", type=str, default="finetuned_checkpoints")
    parser.add_argument("--backbone", type=str, default="prithvi_eo_v2_600_tl")
    parser.add_argument("--epochs", type=int, default=50)

    # 用默认列表没问题，但命令行传参想改的话建议用 --bands JSON；这里先保留默认
    parser.add_argument("--bands", dest="BANDS", default=['COASTAL AEROSOL','BLUE','GREEN','RED','RED_EDGE_1','RED_EDGE_2','RED_EDGE_3','NIR_BROAD','WATER_VAPOR','CIRRUS','SWIR_1','SWIR_2','SLOPE','DEM'])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1) 
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # 布尔参数用 flag 方式
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--plot_on_val", action="store_true")
    parser.add_argument("--device", type=list, default=[0,1,2,3]) 

    return parser.parse_args()

def main():
    cfg = get_args()

    log_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    os.makedirs(log_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="landslide"
    )

    checkpoint_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        dirpath=checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1
    )

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy="ddp_find_unused_parameters_true",
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
    
    val_test_tfms = [
        A.Resize(224, 224),
        A.pytorch.transforms.ToTensorV2(),
    ]
    train_tfms = [
        A.HorizontalFlip(),
        A.Resize(224, 224),
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


    backbone_args = dict(
        backbone=cfg.backbone, 
        backbone_pretrained=False, # False = using local checkpoint
        backbone_ckpt_path=os.path.join(cfg.backbone_path, cfg.backbone, backbone_name),
        backbone_bands=cfg.BANDS,
        backbone_num_frames=1,
    )

    decoder_args = dict(
        decoder="UperNetDecoder", # Unet for Decoder
        decoder_channels=256,
        decoder_scale_modules=True,
    )

    necks = [
        dict(
            name="SelectIndices",
            indices=[7, 15, 23, 31], 
        ),
        dict(name="ReshapeTokensToImage"),
    ]

    model_args = dict(
        **backbone_args,
        **decoder_args,
        num_classes=2,
        head_dropout=cfg.dropout,
        head_channel_list=[128, 64],
        necks=necks,
        rescale=True,
    )

    model = SemanticSegmentationTask(
        model_args=model_args,
        plot_on_val=cfg.plot_on_val,
        loss="focal",
        lr=cfg.lr,
        optimizer="AdamW",
        scheduler="StepLR",
        scheduler_hparams={"step_size": 10, "gamma": 0.9},
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
        print(f"test results is: {test_results}")
        print(f"Test results saved to: {results_path}")


if __name__ == "__main__":
    main()
