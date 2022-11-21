ddp ./lm-main.py \
  --flagfile=flags/mlm3 \
  --hug=all-mpnet-base-v2 \
  $*
