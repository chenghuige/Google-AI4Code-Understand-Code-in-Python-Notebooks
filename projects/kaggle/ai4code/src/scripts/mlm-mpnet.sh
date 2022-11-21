ddp ./lm-main.py \
  --flagfile=flags/mlm \
  --hug=all-mpnet-base-v2 \
  $*
