#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   changekeyvalue2key.py
#        \author   chenghuige  
#          \date   2010-12-09 11:55:08.860840
#   \Description   <key,value> -> key   
# ==============================================================================

import sys

def run(input, output):
  out = open(output, 'w')
  for line in open(input):
    line = line.strip()
    l = line.split()
    if (len(l) > 0):
      out.write(l[0]+'\n')

if __name__ == '__main__':
  input =''
  output = ''
  if (len(sys.argv) > 1):
    input = sys.argv[1]
  if (len(sys.argv) > 2):
    output = sys.argv[2]
  run(input, output)
