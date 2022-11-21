#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   tlc2malloc.py
#        \author   chenghuige  
#          \date   2014-03-27 11:14:57.270128
#   \Description  
# ==============================================================================

import sys,os

num = 0
for line in open(sys.argv[1]):
	num += 1
	if num == 1:
		continue
	l = line.strip().split('\t')
	l2 = [l[1]]
	for i in range(2, len(l)):
		l2.append("%d:%s"%(i - 1, l[i]))
	print '\t'.join(l2)
 
