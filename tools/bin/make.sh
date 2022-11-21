time make -j22 2>&1 $1 | tee make-result.txt
grep --color=auto -B 1 -A 3 error:  make-result.txt
