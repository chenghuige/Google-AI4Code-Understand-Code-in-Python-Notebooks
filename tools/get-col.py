#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================

import sys,os

col = int(sys.argv[1])
for line in sys.stdin:
  print(line.strip().split('\t')[col])

