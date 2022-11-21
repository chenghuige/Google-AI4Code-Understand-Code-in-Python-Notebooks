#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   remove_space.py
#        \author   chenghuige  
#          \date   2010-11-07 20:16:20.157547
#   \Description  
# ==============================================================================

import sys

def run(input, output):
  out = open(output, 'w')
  for line in open(input):
    out.write(''.join(line.split()))
    out.write('\n')
  out.close()
 
if __name__ == '__main__':
  input = sys.argv[1]
  output = sys.argv[2]
  run(input, output)
  
