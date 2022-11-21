i=$1
ddp ./main.py \
  --flagfile=flags/context4-2-d-s-ext-pre_emlm_mlm \
  --external=ext_100000_$i \
  --external_idx=$i \
  --num_workers=8 \
  --ep=$((i+1)) 

