ddp ./lm-main.py \
  --flagfile=flags/mlm2 \
  --hug=all-mpnet-base-v2 \
  $*
