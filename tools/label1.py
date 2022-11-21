#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   label1.py
#        \author   chenghuige  
#          \date   2014-01-04 20:36:29.412104
#   \Description  
# ==============================================================================

import sys,os

out = open(sys.argv[2],'w')
for line in open(sys.argv[1]):
    l = line.strip().split()
    out.write("%s\t1\n"%l[0])


 
