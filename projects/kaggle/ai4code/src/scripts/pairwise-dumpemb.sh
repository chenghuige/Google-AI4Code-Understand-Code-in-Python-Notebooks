ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.dump_emb \
  --restore_configs \
  --mode=valid \
  --eval_name=eval \
  --dump_emb \
  --save_emb \
  --save_final=0 \
  --add_end_source \
  --save_probs \
  $*

