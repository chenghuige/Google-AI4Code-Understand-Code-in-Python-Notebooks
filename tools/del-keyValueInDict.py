#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   delkeyvalueindict.py
#        \author   chenghuige  
#          \date   2010-12-09 11:55:08.860840
#   \Description   input1里面删除input2里面的词,value  
# ==============================================================================

import sys, collections

#add input2 to input1 the result is input1
def merge(input1, input2):
  d = {}
  d2 = collections.defaultdict(lambda:0)
  for line in open(input1):
    line = line.strip()
    l = line.split()
    d[l[0]] = l[1]
    d2[l[0]] += 200
  for line in open(input2):
    line = line.strip()
    d2[line] += 1 
  out = open(input1, 'w')
  l = sorted(d2.items(), key=lambda d2:d[0])
  for k,v in l:
    if (v == 200):
      out.write(k + ' ' + d[k] + '\n')

if __name__ == '__main__':
  input1 = 'ambiguity.txt'
  input2 = 'add_common.txt'
  if (len(sys.argv) > 1):
    input1 = sys.argv[1]
  if (len(sys.argv) > 2):
    input2 = sys.argv[2]
  merge(input1, input2)
