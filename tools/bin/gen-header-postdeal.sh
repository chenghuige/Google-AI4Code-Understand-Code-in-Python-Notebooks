cat $1 | fix-enum-class.py > $1.cc 
mv $1.cc  $1
