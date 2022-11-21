#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================

import sys,os

col_name = sys.argv[2]
col = 0

infile = open(sys.argv[1]) 
col = infile.readline().split().index(col_name)

if len(sys.argv) > 3:
	out = open(sys.argv[3], 'w')
	for line in infile:
		out.write("%s\n" % line.split()[col])
else:
	for line in infile:
		print line.strip().split()[col]


 
