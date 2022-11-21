#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   show-trate-feature.py
#        \author   chenghuige  
#          \date   2014-04-25 16:50:55.237400
#   \Description  
# ==============================================================================

from gezi import *

identifer = DoubleIdentifer()
identifer.Load(sys.argv[2])

total = identifer.size()
num = 0
for line in open(sys.argv[1]):
    num += 1
    if (num < 3):
        continue 
    l = line.strip().split() 
    if num == 3:
        print 'bias:' + '\t' + l[-1]
        continue
    id = int(l[0]) 
    if id < total:
        print identifer.key(id) + '\t' + l[-1]
    else:
        print 'c:' + identifer.key(id - total) + '\t' + l[-1]
