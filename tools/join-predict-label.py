#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   join-predict-label.py
#        \author   chenghuige  
#          \date   2014-11-06 16:11:02.975548
#   \Description  
# ==============================================================================

import sys,os

predicts = [line.strip() for line in open(sys.argv[1])] 
labels = [line.split()[0] for line in open(sys.argv[2])]

if len(labels) > len(predicts):
	labels = labels[1:]

for i in range(len(predicts)):
	print labels[i],'\t',predicts[i]
