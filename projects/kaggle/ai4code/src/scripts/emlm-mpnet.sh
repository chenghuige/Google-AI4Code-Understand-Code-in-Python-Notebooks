i=$1
ddp ./lm-main.py \
  --flagfile=flags/emlm \
  --external=ext_100000_$1 \
  --hug=all-mpnet-base-v2 \
  --ep=$((i+1)) 
