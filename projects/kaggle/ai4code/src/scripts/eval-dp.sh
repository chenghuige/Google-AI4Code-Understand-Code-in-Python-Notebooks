./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=all-mpnet-base-v2.flag-pairwise13-2.pooling_mask-attention_mask \
  --num_negs=2 \
  $*

