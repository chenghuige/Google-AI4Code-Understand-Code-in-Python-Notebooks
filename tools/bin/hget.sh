mkdir -p $2
hadoop fs -get $1/*0 $2 &
hadoop fs -get $1/*1 $2 &
hadoop fs -get $1/*2 $2 &
hadoop fs -get $1/*3 $2 &
hadoop fs -get $1/*4 $2 &
hadoop fs -get $1/*5 $2 &
hadoop fs -get $1/*6 $2 &
hadoop fs -get $1/*7 $2 &
hadoop fs -get $1/*8 $2 &
hadoop fs -get $1/*9 $2 &
