#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================

import sys,os

col_names = sys.argv[1].split(',')

first = True
for line in sys.stdin:
  if first:
    cols = infile.readline().rstrip('\n').split('\t')
    selected = [idx for idx in range(len(cols) if cols[idx] in col_names]
    first = False
  l = line.rstrip('\n').split('\t')
  print '\t'.join([l[idx] for idx in selected])
 
