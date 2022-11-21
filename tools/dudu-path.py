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

path = Popen('pwd', shell = True, stdout = PIPE).stdout.readline().strip().decode('utf8')

path =  'featurize@10.0.139.185:' + path

if (len(sys.argv) > 1):
	path += '/' + sys.argv[1]

print(path)
print(f'rsync -avP {path} .')
