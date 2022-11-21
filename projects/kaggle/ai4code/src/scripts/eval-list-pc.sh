./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval-pc \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=deberta-v3-small.flag-pairwise14-4-cat-insert-oof-ft.eval \
  --num_negs=2 \
  --list_infer \
  --max_len=160 \
  --max_context_len=30 \
  $*

