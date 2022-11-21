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

path = sys.argv[1] + ':' + path.decode()

if len(sys.argv) > 2:
  path += '/' + sys.argv[2]
else:
  path += '/*'

command = 'rsync -avP %s .' % path 
print(command)
os.system(command)
 
