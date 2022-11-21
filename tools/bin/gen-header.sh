#h2cc.py -a $1 $1.cc 

cat $1 | fix-c++11.py | fix-constructor.py > $1.cc 
mv $1.cc  $1
