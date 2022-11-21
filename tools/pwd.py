#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   pwd.py
#        \author   chenghuige  
#          \date   2014-02-06 21:29:30.994793
#   \Description  
# ==============================================================================

import sys,os
from subprocess import *

os.system('pwd')
path = Popen('pwd', shell = True, stdout = PIPE).stdout.readline().strip()

path = path.decode()

if (len(sys.argv) > 1):
  path += "/" + sys.argv[1]

print(path)
