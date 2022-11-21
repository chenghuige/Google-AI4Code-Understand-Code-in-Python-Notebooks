ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval \
  --restore_configs \
  --wandb_scratch \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=pmminilm.flag-pairwise14-2 \
  --num_negs=2 \
  --list_infer \
  --max_len=128 \
  --max_context_len=25 \
  --list_leak=0 \
  --list_train_ordered=0 \
  $*

