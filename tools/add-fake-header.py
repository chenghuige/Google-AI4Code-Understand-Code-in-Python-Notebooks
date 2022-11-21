#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   add-fake-header.py
#        \author   chenghuige  
#          \date   2014-10-08 15:02:56.496216
#   \Description  
# ==============================================================================

import sys,os

num_cols = len(open(sys.argv[1]).readline().split())
start = 1
if len(sys.argv) > 2:
	start = int(sys.argv[2])

l = ['t{}'.format(i) for i in range(start)]

for i in range(num_cols - start):
	l.append('f{}'.format(i))

l[0] = '#%s'%l[0] 
print '\t'.join(l)

for line in open(sys.argv[1]):
	print line,

 
