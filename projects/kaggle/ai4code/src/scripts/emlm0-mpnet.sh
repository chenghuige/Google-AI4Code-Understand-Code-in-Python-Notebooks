ddp ./lm-main.py \
  --flagfile=flags/emlm \
  --external=ext_100000_0 \
  --hug=all-mpnet-base-v2 \
  $*
