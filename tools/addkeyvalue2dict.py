#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   addkeyvalue2dict.py
#        \author   chenghuige  
#          \date   2010-12-09 11:55:08.860840
#   \Description   像input1里面添加input2里面的词,value  
# ==============================================================================

import sys

#add input2 to input1 the result is input1
def merge(input1, input2):
  d = {}
  for line in open(input1):
    line = line.strip()
    l = line.split()
    d[l[0]] = l[1]
  for line in open(input2):
    line = line.strip()
    l = line.split()
    d[l[0]] = l[1] 
  out = open(input1, 'w')
  l = sorted(d.items(), key=lambda d:d[0])
  for k,v in l:
    out.write(k + ' ' + v + '\n')

if __name__ == '__main__':
  input1 = 'ambiguity.txt'
  input2 = 'add_common.txt'
  if (len(sys.argv) > 1):
    input1 = sys.argv[1]
  if (len(sys.argv) > 2):
    input2 = sys.argv[2]
  merge(input1, input2)
