 perl -F"\t" -ane 'if($_=~/proctime:total:(\d+)/ and $1 > -1) {print "$1\n";}' $1 | perl -MList::Util=min -lane '{push @a,$F[0]}END{print min @a}'
