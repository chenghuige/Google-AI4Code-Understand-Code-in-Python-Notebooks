#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   melt2csv.py
#        \author   chenghuige  
#          \date   2014-10-02 11:57:51.929779
#   \Description  
# ==============================================================================

import sys,os

numCols = 0
for line in open(sys.argv[1]):
	if line.startswith('#') or line.startswith('_'):
		line = line[1:]
	numColsNow = len(line.split('\t'))
	if numColsNow > numCols:
		numCols = numColsNow
	elif numColsNow < numCols:
		#print numColsNow, numCols
		raise 'Error:', numColsNow, numCols
		continue
	print line,

 
