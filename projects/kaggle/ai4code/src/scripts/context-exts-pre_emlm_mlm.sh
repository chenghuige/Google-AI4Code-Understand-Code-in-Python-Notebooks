max=10
for (( i=$1; i < $max; ++i ))
do
  echo "$i"
  ./scripts/context-ext-pre_emlm_mlm.sh $i
done
