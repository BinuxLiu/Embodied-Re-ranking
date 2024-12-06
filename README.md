# Embodied Re-ranking

## shell

```shell
# ER
python eval.py --aggregation g2m --backbone dinov2_vitb14 --dataset_name pitts30k --infer_batch_size 128 --resize_test_imgs --resize 224 224 --resume ../checkpoints/G2M_GPMS.pth --use_ca --num_hiddens 64 --ranking er
# ER-Net
python train_er.py --aggregation g2m --backbone dinov2_vitb14 --dataset_name pitts30k --infer_batch_size 128 --resize_test_imgs --resize 224 224 --resume ../checkpoints/G2M_GPMS.pth --use_ca --num_hiddens 64 --epochs_num 30
```