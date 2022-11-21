ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval-ext \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=all-mpnet-base-v2.flag-pairwise14-2-ext-pre_emlm_mlm \
  --num_negs=2 \
  $*

