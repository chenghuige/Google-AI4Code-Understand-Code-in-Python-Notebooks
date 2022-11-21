#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   /home/users/chenghuige/tools/gen-boost-seralize.py
#        \author   chenghuige  
#          \date   2014-09-07 16:42:04.643648
#   \Description  
# ==============================================================================

import sys,os

for line in sys.stdin:
	line = line.strip()
	if len(line) == 0 or line.startswith('//'):
		continue
	print 'fmt::print_line("{}");'.format(line.replace('"',r'\"'))

 
