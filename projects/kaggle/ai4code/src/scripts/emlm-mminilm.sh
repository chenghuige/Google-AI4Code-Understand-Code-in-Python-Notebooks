i=$1
ddp ./lm-main.py \
  --flagfile=flags/emlm \
  --external=ext_100000_$((i+10)) \
  --hug=paraphrase-multilingual-MiniLM-L12-v2 \
  --ep=$((i+1)) 
