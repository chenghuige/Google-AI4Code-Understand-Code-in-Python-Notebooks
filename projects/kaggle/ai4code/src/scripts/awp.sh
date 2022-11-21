ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.awp \
  --restore_configs \
  --awp_train \
  --adv_start_epoch=1 \
  $*

