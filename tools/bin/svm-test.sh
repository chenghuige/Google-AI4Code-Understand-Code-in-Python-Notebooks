svm-scale -l 0 -r $2.range $1 > $1.scale 
echo "finish scale"
svm-predict -b 1 $1.scale $2.model $1.predict
echo "finish predict"
svm-evaluate.sh $1
