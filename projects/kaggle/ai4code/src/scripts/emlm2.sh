ddp ./lm-main.py \
  --flagfile=flags/emlm2 \
  --external=ext_100000_$1 \
  $*
