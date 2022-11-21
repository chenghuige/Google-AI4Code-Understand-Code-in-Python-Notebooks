#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   pack-all.py
#        \author   chenghuige  
#          \date   2014-03-08 13:18:42.098915
#   \Description  
# ==============================================================================

import sys,os
import glob 

l = []
os.system('rm -rf ./dist')
if (len(sys.argv) <= 1):
	l = glob.glob('*.py')
else:
	l = sys.argv[1:]
for file_ in l:
	print file_ 
	os.system('cxfreeze %s'%file_)
os.system('pack.sh')
 
