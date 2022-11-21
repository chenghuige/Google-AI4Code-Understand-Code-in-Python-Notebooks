mkdir -p $2
rsync -avP $1/model*ckpt* $1/checkpoint $2
