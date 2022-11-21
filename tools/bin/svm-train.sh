svm-scale.sh $1
svm-train -m 1024 -b 1 $1.scale $1.model
