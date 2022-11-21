./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.pairwise_eval \
  --restore_configs \
  --mode=valid \
  --pairwise_eval \
  --eval_name=eval \
  --save_emb=0 \
  --save_final \
  $*

