#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os


indexes = [int(x) for x in sys.argv[1].split(',')]

for line in sys.stdin:
  l = line.rstrip('\n').split()
  print('\t'.join(l))
