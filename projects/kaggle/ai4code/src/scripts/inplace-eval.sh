ddp ./main.py \
  --pretrain_restart=0 \
  --mn=$1 \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  $*

