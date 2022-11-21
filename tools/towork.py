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

path =  path.decode().replace('data', 'work')

input = sys.argv[1]

command = 'rsync -avP --delete %s %s' % (input, path) 
print(command)
os.system(command)
 
