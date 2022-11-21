#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   merge-tlc.py
#        \author   chenghuige  
#          \date   2014-01-11 21:36:02.564392
#   \Description  
# ==============================================================================

import sys,os

t1 = open(sys.argv[1]).readlines()

t2 = open(sys.argv[2]).readlines()[1:]

t = t1 + t2
 
out = open(sys.argv[3], 'w')
for line in t:
	out.write(line)
	
