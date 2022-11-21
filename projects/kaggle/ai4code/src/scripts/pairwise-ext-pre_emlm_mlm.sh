i=$1
ddp ./main.py \
  --flagfile=flags/pairwise14-2-ext-pre_emlm_mlm \
  --external=ext_100000_$i \
  --external_idx=$i \
  --ep=$((i+1)) 

