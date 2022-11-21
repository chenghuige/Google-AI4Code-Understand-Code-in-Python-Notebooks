#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   talbe_info.py
#        \author   chenghuige  
#          \date   2012-06-14 20:28:59.213575
#   \Description  
# ==============================================================================

import sys,os

rows = 0;
cols = 0;
line1 =""
for line in open(sys.argv[1]):
	if (rows == 0):
		line1 = line
		l = line.split('\t')
		cols = len(l)
	rows += 1

print "cols: %d\nrows: %d"%(cols,rows)
print line
 
