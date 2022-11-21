#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gen-header-all.py
#        \author   chenghuige  
#          \date   2014-04-17 07:42:37.239655
#   \Description  
# ==============================================================================

import sys,os
import glob 
for file_ in glob.glob('*.h'):
	cmd = 'sh gen-header.sh ' + file_ 
	print cmd
	os.system(cmd)
 
