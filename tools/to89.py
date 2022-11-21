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

path = Popen('pwd', shell = True, stdout = PIPE).stdout.readline().strip()

path =  '10.160.58.89:' + path.decode()

input = sys.argv[1]

command = 'rsync -avP %s %s' % (input, path) 
print(command)
os.system(command)
 
