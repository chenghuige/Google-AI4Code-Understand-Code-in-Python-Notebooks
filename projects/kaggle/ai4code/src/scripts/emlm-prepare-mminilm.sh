./lm-main.py \
  --flagfile=flags/emlm \
  --hug=paraphrase-multilingual-MiniLM-L12-v2 \
  --external=ext_100000_$1 \
  --prepare \
  $*
