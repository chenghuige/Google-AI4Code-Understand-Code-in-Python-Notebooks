#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   rand.py
#        \author   chenghuige  
#          \date   2013-12-12 16:47:53.081976
#   \Description  
# ==============================================================================

import sys
import random

if __name__ == '__main__':
  with open(sys.argv[1], 'r') as f:
    flist = f.readlines()
    #flist2 = []
    if (len(sys.argv) > 2):
      flist = flist[:int(sys.argv[2])]
      #flist2 = flist[:int(sys.argv[2])]
      #flist = flist[int(sys.argv[2]):]
    random.shuffle(flist)
    
    #for line in flist2:
    #  print line.strip()
    for line in flist:
      print(line.strip())


 
