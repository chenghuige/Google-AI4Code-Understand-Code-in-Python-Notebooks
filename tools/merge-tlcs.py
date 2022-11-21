#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   merge-tlc.py
#        \author   chenghuige  
#          \date   2014-01-11 21:36:02.564392
#   \Description  
# ==============================================================================

import sys,os

l = open(sys.argv[1]).readlines()

for i in range(2, len(sys.argv) - 1):
	l += open(sys.argv[i]).readlines()[1:]
 
out = open(sys.argv[-1], 'w')
for line in l:
	out.write(line)
	
