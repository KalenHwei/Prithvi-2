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

from terratorch import BACKBONE_REGISTRY

from PrithviWxC.model import PrithviWxC
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)
import yaml # 确保在文件顶部添加这个 import

from peft import LoraConfig, get_peft_model

# MoE Imports
# from MoE import inject_moe, patch_training_step_with_moe
from deepseekMoE import inject_moe, patch_training_step_with_moe

class DualPrithviEncoder(torch.nn.Module):
    """
    Final corrected version of the DualPrithviEncoder class.
    """
    def __init__(self, backbone1_args: dict, backbone2_args: dict, all_bands: list):
        super().__init__()

        # --- Part 1: Create Terratorch (EO) model (no changes) ---
        def build_eo_from_registry(args_dict):
            # ... (this inner function is correct and remains unchanged)
            args = args_dict.copy()
            model_name = args.pop("backbone")
            build_kwargs = {}
            for key, value in args.items():
                if key.startswith("backbone_"):
                    new_key = key.replace("backbone_", "", 1)
                    build_kwargs[new_key] = value
                else:
                    build_kwargs[key] = value
            return BACKBONE_REGISTRY.build(model_name, **build_kwargs)

        print("信息: 正在通过 Terratorch Registry 创建 Backbone 1 (EO)...")
        self.backbone1 = build_eo_from_registry(backbone1_args)
        
        # --- Part 2: Create Prithvi-WxC model using config.yaml and providing all required args ---
        print("信息: 正在手动创建 Backbone 2 (WxC)...")
        
        # 1. Load config.yaml
        wxc_ckpt_path = backbone2_args["backbone_ckpt_path"]
        config_path = os.path.join(os.path.dirname(wxc_ckpt_path), "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"WxC model's config.yaml not found at: {config_path}")
        
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        model_args = {
            "in_channels": cfg["in_channels"],
            "input_size_time": cfg["input_size_time"],
            "in_channels_static": cfg["in_channels_static"],
            "input_scalers_mu": in_mu,
            "input_scalers_sigma": in_sig,
            "input_scalers_epsilon": cfg["input_scalers_epsilon"],
            "static_input_scalers_mu": static_mu,
            "static_input_scalers_sigma": static_sig,
            "static_input_scalers_epsilon": cfg["static_input_scalers_epsilon"],
            "output_scalers": output_sig ** 0.5,
            "n_lats_px": cfg["n_lats_px"],
            "n_lons_px": cfg["n_lons_px"],
            "patch_size_px": cfg["patch_size_px"],
            "mask_unit_size_px": cfg["mask_unit_size_px"],
            "mask_ratio_inputs": masking_ratio,
            "mask_ratio_targets": 0.0,
            "embed_dim": cfg["embed_dim"],
            "n_blocks_encoder": cfg["n_blocks_encoder"],
            "n_blocks_decoder": cfg["n_blocks_decoder"],
            "mlp_multiplier": cfg["mlp_multiplier"],
            "n_heads": cfg["n_heads"],
            "dropout": cfg["dropout"],
            "drop_path": cfg["drop_path"],
            "parameter_dropout": cfg["parameter_dropout"],
            "residual": residual,
            "masking_mode": masking_mode,
            "encoder_shifting": encoder_shifting,
            "decoder_shifting": decoder_shifting,
            "positional_encoding": positional_encoding,
            "checkpoint_encoder": [],
            "checkpoint_decoder": [],
        }
        self.backbone2 = PrithviWxC(**model_params)

        # 6. Load the weights
        if os.path.exists(wxc_ckpt_path):
            print(f"信息: 正在从 {wxc_ckpt_path} 加载 WxC 模型权重...")
            state_dict = torch.load(wxc_ckpt_path, map_location='cpu')
            self.backbone2.load_state_dict(state_dict['model_state'])
        else:
            raise FileNotFoundError(f"WxC model's weight file not found: {wxc_ckpt_path}")

        # --- Part 3: Prepare input slicing logic (no changes) ---
        self.bands1 = backbone1_args["backbone_bands"]
        self.bands2 = backbone2_args["backbone_bands"]
        self.indices1 = torch.tensor([all_bands.index(b) for b in self.bands1])
        self.indices2 = torch.tensor([all_bands.index(b) for b in self.bands2])
        
        print("信息: DualPrithviEncoder 初始化完成。")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (forward pass logic remains correct and unchanged)
        self.indices1 = self.indices1.to(x.device)
        self.indices2 = self.indices2.to(x.device)

        x_for_eo = torch.index_select(x, dim=2, index=self.indices1)
        x_for_wxc = torch.index_select(x, dim=2, index=self.indices2)
        
        embedding1 = self.backbone1(x_for_eo)
        embedding2 = self.backbone2(x_for_wxc)

        assert embedding1.shape == embedding2.shape, \
            f"主干网络的输出形状不匹配！ {embedding1.shape} vs {embedding2.shape}"
        
        combined_embedding = embedding1 + embedding2
        return combined_embedding

def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"可训练参数: {trainable_params} || 总参数: {all_param} || 可训练比例: {100 * trainable_params / all_param:.2f}%")

print("print_trainable_parameters这一步成功")

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    # 通用参数
    parser.add_argument("--data_path", type=str, default="/data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/dataset_for_temporal_corp/data")
    parser.add_argument("--log_saving_path", type=str, default="logs")
    parser.add_argument("--checkpoint_saving_path", type=str, default="finetuned_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    # 数据参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--class_weights", type=list, default=[0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462, 1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702])
    
    # 模型参数
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # 主干网络1 (主要) 参数
    parser.add_argument("--backbone1", type=str, default="prithvi_eo_v2_600_tl")
    parser.add_argument("--backbone1_path", type=str, default="checkpoints", help="主干网络1的权重路径。")
    parser.add_argument("--backbone1_bands", dest="BANDS1", nargs='+', default=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"])
   
    # 新增：双主干网络参数
      # 新增：双主干网络参数
    parser.add_argument("--use_dual_backbone", action="store_true", help="使用双主干网络架构。")
    parser.add_argument("--backbone2", type=str, default="prithvi_wxc_2300", help="第二个主干网络的名称。")
    parser.add_argument("--backbone2_path", type=str, default="checkpoints", help="主干网络2的权重路径。")
    parser.add_argument("--backbone2_bands", dest="BANDS2", nargs='+', default=["TCWV", "TCW", "U10", "V10", "T2M", "SP", "MSL", "TCC", "U100", "V100", "T850", "T500", "Z500"], help="第二个主干网络使用的波段列表。")

    # 训练控制
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="使用的GPU设备列表 (例如 '0,1')。")
    
    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning the backbone.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")

    # MoE 参数
    parser.add_argument("--use_moe", action="store_true", help="Initialize MoE instead of FFN")
    parser.add_argument("--moe_n_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=1, choices=[1,2])
    parser.add_argument("--moe_capacity_factor", type=float, default=1.25)
    parser.add_argument("--moe_dropout", type=float, default=0.0)
    parser.add_argument("--moe_select", type=str, default="last_k", choices=["all","last_k"], help="替换哪些层")
    parser.add_argument("--moe_k", type=int, default=4, help="当 select=last_k 时，替换末尾K个 MLP/FFN")
    parser.add_argument("--moe_aux_coef", type=float, default=1e-2)

    parser.add_argument("--exp_name", type=str, default="dual_encoder_exp", help="用于日志和权重的自定义实验名称。")

    return parser.parse_args()

print("args这一步成功")


def main():
    print("进入main成功")

    cfg = get_args()
    pl.seed_everything(cfg.seed)

    print("0")

    if cfg.backbone1 == "prithvi_eo_v2_600_tl":
        backbone_name1 = "Prithvi_EO_V2_600M_TL.pt"
    elif cfg.backbone1 == "prithvi_eo_v2_600":
        backbone_name1 = "Prithvi_EO_V2_600.pt"
    elif cfg.backbone1 == "prithvi_eo_v2_300_tl":
        backbone_name1 = "Prithvi_EO_V2_300_TL.pt"
    elif cfg.backbone1 == "prithvi_eo_v2_300":
        backbone_name1 = "Prithvi_EO_V2_300.pt"
    else:
        raise ValueError(f"backbone {cfg.backbone1} not supported")

    backbone_name2 = None # 默认为空
    if cfg.use_dual_backbone:
        if cfg.backbone2 == "prithvi_wxc_2300": # 【已修正】使用新的、正确的全名进行检查
            # 使用您提供的更准确的文件名
            backbone_name2 = "prithvi.wxc.2300m.v1.pt"
        else:
            raise ValueError(f"backbone2 {cfg.backbone2} not supported")
        
    # 日志和权重保存设置
    base_log_dir = os.path.join(cfg.checkpoint_saving_path, cfg.log_saving_path)
    logger = TensorBoardLogger(save_dir=base_log_dir, name="multi-temporal-crop", version=cfg.exp_name)
    checkpoint_callback = ModelCheckpoint(monitor="val/mIoU", mode="max", dirpath=logger.log_dir, filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}", save_top_k=1)
    
    print("1")
    # 数据模块设置
    if cfg.BANDS2 is None:
        all_bands = cfg.BANDS1
    else:
        all_bands = sorted(list(set(cfg.BANDS1 + cfg.BANDS2)))
    print(f"信息: Dataloader将加载以下波段: {all_bands}")
    
    
    train_transforms = [
        terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
        A.Flip(),
        A.pytorch.transforms.ToTensorV2(),
        terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=cfg.num_frames),
    ]
    data_module = MultiTemporalCropClassificationDataModule(
        batch_size=cfg.batch_size,
        data_root=cfg.data_path,
        train_transform=train_transforms,
        reduce_zero_label=True,
        expand_temporal_dimension=True,
        num_workers=7,
        use_metadata=True,
    )
    

    # 定义通用的解码器和neck参数
    decoder_args = dict(decoder="UperNetDecoder", decoder_channels=256, decoder_scale_modules=True)
    necks = [dict(name="SelectIndices", indices=[7, 15, 23, 31]), dict(name="ReshapeTokensToImage", effective_time_dim=cfg.num_frames)]

    model_to_train = None

    print("2")

    if cfg.use_dual_backbone:
        print("信息: 正在构建双主干网络模型...")
        # 主干网络1的配置
        backbone1_args = dict(
            backbone=cfg.backbone1, 
            # 【已修正】设置为 True 以加载本地权重
            backbone_pretrained=True, 
            backbone_ckpt_path=os.path.join(cfg.backbone1_path, cfg.backbone1, backbone_name1),
            backbone_coords_encoding=["time", "location"], 
            backbone_bands=cfg.BANDS1, 
            backbone_num_frames=cfg.num_frames
        )

        # 主干网络2的配置
        backbone2_args = dict(
            backbone=cfg.backbone2, 
            # 【已修正】设置为 True 以加载本地权重
            backbone_pretrained=True,
            backbone_ckpt_path=os.path.join(cfg.backbone2_path, cfg.backbone2, backbone_name2),
            backbone_coords_encoding=["time", "location"], 
            backbone_bands=cfg.BANDS2, 
            backbone_num_frames=cfg.num_frames
        )

        # 创建自定义的双编码器
        custom_encoder = DualPrithviEncoder(backbone1_args, backbone2_args, all_bands=all_bands)
        
        # 创建解码器
        decoder = terratorch.models.get_model(**decoder_args)

        # 手动构建完整的EncoderDecoder模型
        model_to_train = terratorch.models.EncoderDecoder(
            encoder=custom_encoder, decoder=decoder, necks=necks,
            head_dropout=cfg.dropout, num_classes=len(cfg.class_weights), rescale=True
        )

    else:
        print("信息: 正在构建单主干网络模型...")
        # 单主干网络逻辑
        backbone_args = dict(
            backbone=cfg.backbone1, 
            # 【已修正】单主干网络模式下同样需要设置为 True
            backbone_pretrained=True,
            backbone_ckpt_path=os.path.join(cfg.backbone1_path, cfg.backbone1, backbone_name1),
            backbone_coords_encoding=["time", "location"], 
            backbone_bands=cfg.BANDS1, 
            backbone_num_frames=cfg.num_frames,
        )
        model_args = dict(**backbone_args, **decoder_args, num_classes=len(cfg.class_weights),
                          head_dropout=cfg.dropout, necks=necks, rescale=True)


    print("3")


    # 实例化 Lightning Task
    task = SemanticSegmentationTask(
        model=model_to_train, 
        model_args=None if model_to_train else model_args,
        model_factory=None if model_to_train else "EncoderDecoderFactory",
        plot_on_val=False, class_weights=cfg.class_weights, loss="ce", lr=cfg.lr, optimizer="AdamW",
        optimizer_hparams=dict(weight_decay=cfg.weight_decay), ignore_index=-1,
        freeze_backbone=cfg.freeze_backbone, freeze_decoder=False
    )
    # ==============================================================================

    print("4")

    # 如果启用，则应用LoRA (逻辑不变)
    if not cfg.freeze_backbone and cfg.use_lora:
        print("信息: LoRA已启用。正在向主干网络应用LoRA...")
        lora_config = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", target_modules=["qkv", "proj"])
        
        if cfg.use_dual_backbone:
            task.model.encoder.backbone1 = get_peft_model(task.model.encoder.backbone1, lora_config)
            task.model.encoder.backbone2 = get_peft_model(task.model.encoder.backbone2, lora_config)
            print("信息: LoRA 已应用到两个主干网络。")
        else:
            task.model.encoder = get_peft_model(task.model.encoder, lora_config)
            print("信息: LoRA 已应用到单个主干网络。")
        
        print("信息: LoRA应用后可训练参数概览:")
        print_trainable_parameters(task)

    print("5")

    # Trainer设置与执行 (逻辑不变)
    trainer = pl.Trainer(accelerator="cuda", devices=cfg.device, precision="bf1d16-mixed", logger=logger,
                         max_epochs=cfg.epochs, callbacks=[checkpoint_callback], log_every_n_steps=10)
    trainer.fit(task, datamodule=data_module)
    test_results = trainer.test(task, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    if test_results:
        results_path = os.path.join(logger.log_dir, "test_results.json")
        with open(results_path, 'w') as f: json.dump(test_results[0], f, indent=4)
        print(f"测试结果已保存至: {results_path}")


if __name__ == "__main__":
    main()

"""

python your_script_name.py \
  --use_dual_backbone \
  --backbone1 prithvi_eo_v2_600_tl \
  --backbone1_path ./checkpoints \
  --backbone2 prithvi_wxc_2300 \
  --backbone2_path ./checkpoints_wxc \
  --batch_size 8 \
  --device 0,1 \
  --exp_name "我的第一个双网络实验" \
  # ... 其他参数，例如 --use_lora 或 --use_moe


"""