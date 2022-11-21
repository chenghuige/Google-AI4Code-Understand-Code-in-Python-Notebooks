ddp ./lm-main.py \
  --flagfile=flags/mlm4 \
  --hug=all-mpnet-base-v2 \
  $*
