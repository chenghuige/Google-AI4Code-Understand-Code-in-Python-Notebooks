for (( i=0; i<5; i++ ))
do
  ddp ./main.py --allnew --fold=$i --flagfile=$1 --mode=eval $*
done
ddp ./main.py --allnew --online --flagfile=$1 --mode=eval $*
