ddp ./lm-main.py \
  --flagfile=flags/mlm5 \
  --hug=all-mpnet-base-v2 \
  $*
