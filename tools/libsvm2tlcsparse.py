#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   libsvm2tlc.py
#        \author   chenghuige  
#          \date   2014-01-09 11:26:02.819580
#   \Description  
# ==============================================================================

import sys,os

max_index = 1
for line in open(sys.argv[1]):
  l = line.strip().split()
  for item in l[1:]:
    index, value = item.split(':')
    index = int(index)
    if index > max_index:
      max_index = index

num_features = max_index

for line in open(sys.argv[1]):
  l = line.strip().split()
  label = l[0]
  print label,
  for item in l[1:]:
    index, value = item.split(':') 
    print '\t{}\t{}:{}'.format(num_features, int(index) - 1, value),
  print '\n',
	
