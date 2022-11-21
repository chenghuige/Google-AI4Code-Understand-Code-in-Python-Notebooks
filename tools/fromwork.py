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

if len(sys.argv) > 1:
  path += '/' + sys.argv[1]
else:
  path += '/*'

command = 'rsync -avP --delete %s .' % path 
print(command)
os.system(command)
 
