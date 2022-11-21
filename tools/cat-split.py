#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   split.py
#        \author   chenghuige  
#          \date   2016-07-21 13:35:01.560641
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys,os

file = sys.argv[1]
out = sys.argv[2]
count = 12
if len(sys.argv) > 3:
  count = int(sys.argv[3])
command = "cat %s | awk "%file + "'{print $0 >" + '("{}_" int(NR%{}'.format(out, count) + "))}'"
print command 
os.system(command)
  
