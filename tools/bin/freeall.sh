#!/usr/bin/expect -f 
set timeout 1 
spawn sudo chmod 777 -R * 
expect "*assword:*"  
send "run\r"  
expect eof
interact  
