#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   add_newword.py
#        \author   chenghuige  
#          \date   2011-05-24 10:57:49.046864
#   \Description  
# ==============================================================================

import sys

def run(file):
  fman = open('worddict.man','r')
  result = fman.readlines()
  f = open(file,'r')
  index = 1
  for line in f:
    word = line.strip()
    add = '-N [%s] [0(%s)] [] -n 1000\n'%(word, word)
    result.insert(index, add)
    index = index + 1
  fman.close()
  out = open('worddict.man', 'w')
  out.write(''.join(result))
 
if __name__ == '__main__':
    run(sys.argv[1]) 
