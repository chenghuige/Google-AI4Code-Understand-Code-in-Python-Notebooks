ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --save_probs \
  --context_valid_aug=0 \
  --pairwise_dir=all-mpnet-base-v2.flag-pairwise14-2-pre_mlm3.eval \
  --num_negs=2 \
  --list_infer \
  --max_len=168 \
  --max_context_len=32 \
  --list_leak=0 \
  --list_train_ordered=0 \
  $*

