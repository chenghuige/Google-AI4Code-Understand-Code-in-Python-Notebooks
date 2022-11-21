#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   duplicate-feature.py
#        \author   chenghuige  
#          \date   2014-03-15 19:53:12.185030
#   \Description  
# ==============================================================================

import sys,os

last = ''
for line in open(sys.argv[1]):
	print line,
	last = line 

for i in xrange(100):
	print last,
