import os
import json
import argparse

import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch

from terratorch.datamodules import Landslide4SenseNonGeoDataModule

from itransformer import ITransformerSegTask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset_for_landslide/data")
    parser.add_argument("--log_saving_path", type=str, default="logs")
    parser.add_argument("--checkpoint_saving_path", type=str, default="finetuned_checkpoints")

    # iTransformer hyperparams
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 4, 2], help="blocks per stage")
    parser.add_argument("--heads", type=int, nargs=4, default=[4, 4, 8, 8], help="attention heads per stage")
    parser.add_argument("--patch_size", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bands", dest="BANDS", default=['COASTAL AEROSOL','BLUE','GREEN','RED','RED_EDGE_1','RED_EDGE_2','RED_EDGE_3','NIR_BROAD','WATER_VAPOR','CIRRUS','SWIR_1','SWIR_2','SLOPE','DEM'])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--plot_on_val", action="store_true")  # kept for parity; unused here
    parser.add_argument("--device", type=list, default=[0,1,2,3])

    parser.add_argument("--loss", type=str, choices=["focal", "ce"], default="focal")
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)

    parser.add_argument("--num_classes", type=int, default=2)

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
        save_top_k=1,
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

    in_chans = len(cfg.BANDS)

    model = ITransformerSegTask(
        in_chans=in_chans,
        num_classes=cfg.num_classes,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        loss=cfg.loss,
        dropout=cfg.dropout,
        scheduler_step=cfg.scheduler_step,
        scheduler_gamma=cfg.scheduler_gamma,
        embed_dim=cfg.embed_dim,
        depths=tuple(cfg.depths),
        num_heads=tuple(cfg.heads),
        patch_size=cfg.patch_size,
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
    # if freeze backbone:
    # conduct:
    # python run_itransformer.py --freeze_backbone

    # if need warmup:
    # python run_itransformer.py --freeze_backbone --unfreeze_backbone_at 5
    # still need learning_rate decay becaule of the total training parameters are quite lightweighted.