#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get_uniq.py
#        \author   chenghuige  
#          \date   2012-06-29 17:10:58.457199
#   \Description  
# ==============================================================================

import sys,os

pre=''
for line in open(sys.argv[1]):
	line = line.strip().split('\t')[0]
	if (line != pre):
		print line
		pre = line

		

 
