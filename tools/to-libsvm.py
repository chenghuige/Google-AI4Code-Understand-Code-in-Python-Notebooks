#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   to-libsvm.py
#        \author   chenghuige  
#          \date   2014-01-03 22:32:08.826048
#   \Description   table file to libsvm file
# ==============================================================================

import sys,os,math
label_index = 1
if (len(sys.argv) > 2):
    label_index = int(sys.argv[2])
thre = 0.0
if (len(sys.argv) > 3):
    thre = float(sys.argv[3])
idx = 0
for line in open(sys.argv[1]):
    idx += 1
    if (idx == 1):
        continue
    l = line.strip().split('\t')
    l = l[label_index:]
    l2 = []
    for i in range(len(l)):
        if (i == 0):
            l2.append(l[i])
        else:
            val = l[i]
            val_ = float(val)
            if (math.fabs(val_) <= thre):
                continue
            result = "%d:%s"%(i, val)
            l2.append(result.strip())
    print ' '.join(l2)



 
