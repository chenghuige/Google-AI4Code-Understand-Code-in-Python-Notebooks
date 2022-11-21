for (( i=0; i<5; i++ ))
do
  ddp ./main.py --allnew --fold=$i $*
done
ddp ./main.py --allnew --online $*

