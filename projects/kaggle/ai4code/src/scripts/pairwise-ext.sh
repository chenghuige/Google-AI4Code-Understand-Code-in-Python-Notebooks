i=$1
ddp ./main.py \
  --flagfile=flags/pairwise14-2-ext \
  --external=ext_100000_$i \
  --external_idx=$i \
  --ep=$((i+1)) 

