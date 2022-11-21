#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   run-commands-eachline.py
#        \author   chenghuige  
#          \date   2014-11-15 09:36:23.349688
#   \Description  
# ==============================================================================

import sys,os
for line in open(sys.argv[1]):
	command = line.strip()
	os.system(command)
 
