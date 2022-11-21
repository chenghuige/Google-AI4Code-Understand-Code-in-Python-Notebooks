max=10
for (( i=$1; i < $max; ++i ))
do
  echo "$i"
  ./scripts/pairwise-ext-pre_emlm_mlm.sh $i
done
