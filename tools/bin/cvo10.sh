for (( i=0; i<10; i++ ))
do
  ddp ./main.py --allnew --folds=10 --fold=$i $*
done
ddp ./main.py --allnew --online $*

