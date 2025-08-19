
python /data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/scripts/crops/run_dual_lora_moe.py \
  --use_dual_backbone \
  --backbone1 prithvi_eo_v2_600_tl \
  --backbone1_path /data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/checkpoints \
  --backbone2 prithvi_wxc_2300 \
  --backbone2_path /data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/checkpoints \
  --batch_size 2 \
  --device 0,1 \
  --exp_name "我的第一个双网络实验" \
  --use_lora \
  --use_moe \
  --moe_n_experts 8
