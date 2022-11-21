#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   win-path.py
#        \author   chenghuige  
#          \date   2014-02-22 12:04:40.880360
#   \Description  
# ==============================================================================

import sys,os
from subprocess import *

import socket
host = socket.gethostbyname(socket.gethostname())

path = Popen('pwd', shell = True, stdout = PIPE).stdout.readline().strip().decode('utf8')

path =  f'root@{host}:' + path

if (len(sys.argv) > 1):
	path += '/' + sys.argv[1]

print(path)
print(f'rsync -avP {path} .') 
