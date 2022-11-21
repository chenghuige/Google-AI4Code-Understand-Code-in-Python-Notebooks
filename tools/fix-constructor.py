#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   fix-constructor.py
#        \author   chenghuige  
#          \date   2014-04-18 11:08:20.422001
#   \Description  
# ==============================================================================

import sys,os

pre_is_constructor = False
constructor = ''

input = sys.stdin
if len(sys.argv) >  1:
    input = open(sys.argv[1])
for line in input:
	line = line.strip()
	if line.endswith(') :'):
		pre_is_constructor = True
		constructor = line[:line.rfind(':')]
	if not pre_is_constructor:
		print line 
	elif line.endswith(';'):
		pre_is_constructor = False
		print constructor + ';'
	
	

 
