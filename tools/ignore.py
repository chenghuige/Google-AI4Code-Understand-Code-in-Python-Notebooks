#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   ignore.py
#        \author   chenghuige  
#          \date   2014-03-16 21:16:38.168204
#   \Description  
# ==============================================================================

import sys,os

n = 1
if (len(sys.argv) > 2):
	n = int(sys.argv[2]) 
now = 0
for line in open(sys.argv[1]):
	now += 1
	if (now <= n):
		continue
	print line,

 
