#!/bin/bash
# Tool for SSH relation building, v1.0
# wangxinneng, 2006-04-06
# comm comm

#tmp_file for input:"user@host passwd"; tmp_triple for triple edge relation: all keys collected
tmp_file="/tmp/.tmp_file"
tmp_triple="/tmp/.tmp_triple"
tmp_knownhosts="/tmp/.tmp_knownhosts"
if [ -f $tmp_file ];then rm -rf $tmp_file;fi
if [ -f $tmp_triple ];then rm -rf $tmp_triple;fi
if [ -f $tmp_knownhosts ];then rm -rf $tmp_knownhosts;fi

Usage()
{
	cat <<EOH

Tool for SSH relation building.

Usage: `basename $0` [-s|-d|-t] <-l user@host passwd | -f file> [-v|-h|--help]
	-s, --single	Single edge SSH, A2B,A2C,...,A is this host. This is default mode
	-d, --double	Double edge SSH, A2B,B2A,A2C,C2A,...
	-t, --triple	Triple edge SSH, A2B,B2A,A2C,C2A,B2C,C2B,...,more
	-l, --login	Login host and password, only a pair allowed once time. More -l allowed
	-f, --file	Login host and password, a pair each line, i.e."user@host passwd" many lines
	-v|-h, --help	Help message, this message

E.g.	`basename $0` -s -l mp3@testing-app00 mp3passwd -l img@testing-app00 imgpasswd
    	`basename $0` -s -l mp3@testing-app00 mp3passwd -f ./mp3hostlist.txt
    	`basename $0` -t -f ./mp3hostlist.txt
    	`basename $0` -v

mp3hostlist.txt: (old or good ones first, it's useful.) 
mp3@testing-app23 mp3passwd
mp3@testing-app21 mp3passwd

EOH
	exit
}

#default mode is single edge: A2B,A2C
mode_flag=0

if [ $# = 0 ];then Usage;fi
while [ $# != 0 ]
do
	case $1 in
		-s|--single) mode_flag=1;shift;;
		-d|--double) mode_flag=2;shift;;
		-t|--triple) mode_flag=3;shift;;
		-l|--login) if [ "$3" = "" ];then echo "Bad usage";Usage;fi;echo "$2 $3" >> $tmp_file;shift;shift;shift;;
		-f|--file) if [ ! -f $2 ]; then echo "File $2 not exist";Usage;fi;cat $2 >> $tmp_file;shift;shift;;
		-h|-v|-H|--help|-V|--version) Usage;;
		*) echo "Bad usage or wrong argument: $1";echo;Usage;;
	esac
done

#if $tmp_file is empty, show help and exit. Do sort and uniq the input $tmp_file
if [ ! -s $tmp_file ]; then echo "No hosts.";Usage;fi
sort -uk1,1 $tmp_file > $tmp_triple
mv $tmp_triple $tmp_file

#generate ssh key if id_?sa.pub file were not exist
mkdir -p ~/.ssh/
cd ~/.ssh/
if [ ! -f id_dsa.pub ] && [ ! -f id_rsa.pub ]
then
	echo "Press enter"
	ssh-keygen  -q -d -f id_dsa
fi
mykey=`cat ~/.ssh/id_dsa.pub ~/.ssh/id_rsa.pub 2>/dev/null|tail -1`
me=`echo $mykey|cut -d " " -f 3`
myhostname=`echo $me|cut -d "@" -f2`
touch authorized_keys;cat authorize* > tt;sed -e '/$me/d' tt > t;echo $mykey >>t;rm -f authorize* tt;mv t authorized_keys;chmod go-w ..;chmod 600 *;chmod 700 .
ssh $me "echo ssh myself ..."
myknownhosts=`grep -q $myhostname known_hosts|tail -1`

#read $tmp_file for host list and pswd list
id=0
while read target password
do
	host[$id]=$target
	pswd[$id]=$password
	id=$[ id + 1 ]
done<$tmp_file

list_succ=""
list_fail=""
#first round, find the successed and failed list
for((i=0;i<$id;i++))
do
	if [ "${host[$i]}" == "$me" ]; then continue; fi
	hostlist="${host[$i]}|$hostlist"
	ssh $me "ssh ${host[$i]} \"mkdir -p ~/.ssh/;cd ~/.ssh/;touch authorized_keys;cat authorize* > tt;sed -e '/$me/d' tt > t;echo $mykey >>t;rm -f authorize* tt;mv t authorized_keys;chmod go-w ..;chmod 600 *;chmod 700 .;cat known_hosts\"" >>$tmp_knownhosts 2>/dev/null
	if [ $? -eq 0 ]
	then
		list_succ=$list_succ" $i"
	else
		list_fail=$list_fail" $i"
	fi
done
#echo list_succ: [$list_succ] XN 1st round
#echo list_fail: [$list_fail] XN 1st round

list_succagain="$list_succ"
#second round, make it ok by successed hosts. try best to reduce the fail list
while [ 1 ]
do
	list_succnew=""
	count_succnew=0
	for s in $list_succagain
	do
		list_failagain=""
		for f in $list_fail
		do
			ssh ${host[$s]} "ssh ${host[$f]} \"mkdir -p ~/.ssh/;cd ~/.ssh/;touch authorized_keys;cat authorize* > tt;sed -e '/$me/d' tt > t;echo $mykey >>t;rm -f authorize* tt;mv t authorized_keys;chmod go-w ..;chmod 600 *;chmod 700 .;cat known_hosts\"" >>$tmp_knownhosts 2>/dev/null
			if [ $? -eq 0 ]
			then
				list_succ=$list_succ" $f"
				list_succnew=$list_succnew" $f"
				count_succnew=$[ count_succnew + 1 ]
			else
				list_failagain=$list_failagain" $f"
			fi
		done
		list_fail=$list_failagain
	done
	list_succagain=$list_succnew
	if [ $count_succnew -eq 0 ];then break;fi
done
#echo list_succ: [$list_succ] XN 2nd round
#echo list_fail: [$list_fail] XN 2nd round

#last round, password input needed
fail_flag=0
for fl in $list_fail
do
	echo "================================="
	echo "${host[$fl]}'s password: ${pswd[$fl]}"
	echo $mykey|ssh ${host[$fl]} "mkdir -p ~/.ssh/;cd ~/.ssh/;touch authorized_keys;cat authorize* > tt;sed -e '/$me/d' tt > t;cat - >>t;rm -f authorize* tt;mv t authorized_keys;chmod go-w ..;chmod 600 *;chmod 700 .;" 2>/dev/null
	ssh $me "ssh ${host[$fl]} \"cd .ssh/;touch known_hosts;cat known_hosts\"" >>$tmp_knownhosts 2>/dev/null
	if [ $? -eq 0 ]
	then
		list_succ=$list_succ" $fl"
	else
		list_faillast=$list_faillast" ${host[$fl]}"
		fail_flag=$[ fail_flag + 1 ]
	fi
done
cat ~/.ssh/known_hosts >> $tmp_knownhosts
sort -u $tmp_knownhosts > ~/.ssh/known_hosts
if [ $fail_flag -gt 0 ];then echo "FATAL: $fail_flag failed, please check: $list_faillast"; exit;fi
#echo "Single edge done ... XN"

#double edge must do this, maybe triple
if [ $mode_flag -gt 1 ]
then
	for i in $list_succ
	do
		#get the key from the targe for double or triple edge SSH, refresh the known_hosts and correct the permission
		ssh ${host[$i]} "cd .ssh;if [ ! -f id_dsa.pub ] && [ ! -f id_rsa.pub ];then ssh-keygen  -q -d -f id_dsa;fi;touch known_hosts;grep -v $myhostname known_hosts >t;echo $myknownhosts >>t;sort -u t > known_hosts;rm -f t"
		scp ${host[$i]}:~/.ssh/id_?sa.pub /tmp
		cat /tmp/id_dsa.pub /tmp/id_rsa.pub 2>/dev/null|tail -1 >> $tmp_triple
		rm -f /tmp/id_?sa.pub
		tlist="${host[$i]} $tlist"
		elist="${host[$i]}\|$elist"
	done
	hostlist=$hostlist"$me"
	cd ~/.ssh/
	echo $mykey >> $tmp_triple
	grep -vE "$hostlist" authorize* > tt
	sort -u $tmp_triple >> tt
	rm -f authorize* /tmp/id_?sa.pub
	mv tt authorized_keys
	chmod go-w ..;chmod 600 *;chmod 700 .
fi
#echo "Double edge done ... XN"

#triple edge must do this
if [ $mode_flag -eq 3 ]
then
	elist="$elist^$"
	for i in $tlist
	do
		sort -u $tmp_triple|ssh $i "cd ~/.ssh/;touch authorized_keys;cat authorize* > tt;sed -e '/$elist/d' tt > t;cat - >>t;rm -f authorize* tt;mv t authorized_keys;chmod go-w ..;chmod 600 *;chmod 700 ."
		cat $tmp_knownhosts|ssh $i "cd ~/.ssh/;touch known_hosts;cat - >tt;cat known_hosts >>tt;sort -u tt >known_hosts;rm -f tt"
	done
fi
#echo "Triple edge done ... XN"
