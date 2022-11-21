#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   fid2name.py
#        \author   chenghuige  
#          \date   2016-06-25 18:47:35.481474
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys,os

fnames = []
for line in open(sys.argv[1]):
	fname = line.split('#')[0].strip()
	fnames.append(fname)

import re
p = re.compile('f(\d+)')
for line in sys.stdin:
	match = p.search(line)
	if match:
		line = line.replace(match.group(), fnames[int(match.groups()[0])])
		line = '\t'.join(line.split())
		print line 
		continue
	print line,

  
