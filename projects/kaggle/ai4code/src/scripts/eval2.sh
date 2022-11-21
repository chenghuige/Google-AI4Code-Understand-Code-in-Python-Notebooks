ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval2 \
  --restore_configs \
  --wandb_scratch \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=pmminilm.flag-pairwise14-2 \
  --num_negs=4 \
  --n_markdowns=0 \
  $*

