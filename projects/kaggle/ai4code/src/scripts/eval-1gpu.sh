CUDA_VISIBLE_DEVICES=0 ./main.py \
  --gpus=1 \
  --num_workers=0 \
  --pin_mem=0 \
  --persistent_workers=0 \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  $*

