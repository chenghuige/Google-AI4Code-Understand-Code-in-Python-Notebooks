#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================

import sys,os
from prettytable import PrettyTable  


if len(sys.argv) < 2:
  indexes = []
else:
  indexes = [int(x) for x in sys.argv[1].split(',') if x]

sort_index = sys.argv[2]

first = True
for line in sys.stdin:
  l = line.rstrip('\n').split('\t')
  if first:
    first = False 
    if not indexes:
      indexes = range(len(l))
    pt = PrettyTable([str(x) for x in indexes], encoding='gbk')
  #print '\t'.join(l[x] for x in indexes)
  pt.add_row([l[x] for x in indexes])

pt.align = 'l'
print pt.get_string(sortby=(sort_index), reversesort=True).encode('gbk')
