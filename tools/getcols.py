#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================

import sys,os

indexes = [int(x) for x in sys.argv[1].split(',')]

if len(sys.argv) > 2:
  sep = sys.argv[2]
else:
  sep = None

for line in sys.stdin:
  try:
    #l = line.rstrip('\n').split('\t')
    if not sep:
      l = line.rstrip('\n').split()
    else:
      l = line.rstrip('\n').split(sep)
    print('\t'.join([l[x] for x in indexes]))
  except Exception:
    continue
