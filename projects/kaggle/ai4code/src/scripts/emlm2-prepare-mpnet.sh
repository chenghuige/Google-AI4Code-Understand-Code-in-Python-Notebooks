./lm-main.py \
  --flagfile=flags/emlm2 \
  --external=ext_100000_$1 \
  --hug=all-mpnet-base-v2 \
  --prepare \
  $*
