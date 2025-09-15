
# CUDA_VISIBLE_DEVICES=2 nohup python -u aef_rgb_frcnn.py \
#   --variant resnet50_ae_imnet --modality aef --rgb_norm imnet \
#   --train_region uttar_pradesh --in_region uttar_pradesh --oor_regions pak_punjab bangladesh \
#   --image_size 800 --batch_size 8 --epochs 6 \
#   --save_dir runs/aef_resnet50_imnet_s800_e6_b8 \
#   > logs/aef_resnet50_imnet_s800_e6_b8.log 2>&1 &    


CUDA_VISIBLE_DEVICES=3 nohup python -u aef_frcnn.py \
  --variant thin_cnn --modality rgb --rgb_norm imnet \
  --train_region uttar_pradesh --in_region uttar_pradesh --oor_regions pak_punjab bangladesh \
  --image_size 800 --batch_size 8 --epochs 10 \
  --save_dir runs/rgb_thin_cnn_800_e10_b16 \
  > logs/rgb_thin_cnn_800_e10_b16.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -u aef_rgb_frcnn.py \
#   --variant resnet50_ae_imnet --modality rgb --rgb_norm imnet \
#   --train_region uttar_pradesh --in_region uttar_pradesh --oor_regions pak_punjab bangladesh \
#   --image_size 800 --batch_size 8 --epochs 6 \
#   --save_dir runs/rgb_resnet50_imnet_s800_e6_b8 \
#   > logs/rgb_resnet50_imnet_s800_e6_b8.log 2>&1 &