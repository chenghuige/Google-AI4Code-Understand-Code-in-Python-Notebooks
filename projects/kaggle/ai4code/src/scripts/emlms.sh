max=10
for (( i=$1; i < $max; ++i ))
do
  echo "$i"
  ./scripts/emlm.sh $i
done
