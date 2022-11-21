ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval.newrecall \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=all-mpnet-base-v2.flag-pairwise14-2-pre_mlm3.num_negs-9.eval \
  --num_negs=2 \
  --n_markdowns=0 \
  $*

