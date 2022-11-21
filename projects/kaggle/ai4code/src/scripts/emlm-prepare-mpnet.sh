./lm-main.py \
  --flagfile=flags/emlm \
  --external=ext_100000_$1 \
  --prepare \
  --hug=all-mpnet-base-v2 \
  $*
