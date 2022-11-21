#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   label0.py
#        \author   chenghuige  
#          \date   2014-01-04 20:36:27.520345
#   \Description  
# ==============================================================================

import sys,os

out = open(sys.argv[2],'w')
for line in open(sys.argv[1]):
    l = line.strip().split('\t')
    out.write("%s\t0\n"%l[0])


 
