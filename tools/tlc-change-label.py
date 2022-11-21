#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   tlc-change-label.py
#        \author   chenghuige  
#          \date   2014-01-15 07:10:02.225765
#   \Description  
# ==============================================================================

import sys

m = {}
for line in open(sys.argv[1]):
  l = line.strip().split()
  m["_" + l[0]] = l[1]

out = open(sys.argv[3], 'w')
  
line_num = 0
for line in open(sys.argv[2]):
  line_num += 1
  if (line_num == 1):
    out.write(line)
    continue
  l = line.strip().split()
  if l[0] in m:
    l[1] = m[l[0]]
  out.write("%s\n"%"\t".join(l))
 
