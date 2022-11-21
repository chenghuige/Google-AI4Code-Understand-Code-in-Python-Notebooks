#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   touch.py
#        \author   chenghuige  
#          \date   2015-08-01 10:34:34.056678
#   \Description  
# ==============================================================================

import sys,os

filename = sys.argv[1]

os.system('touch %s'%file_name)
os.system('svn add %s'%file_name)
os.system('win-path.py %s'%file_name)
os.system('chmod 777 %s'%file_name)
