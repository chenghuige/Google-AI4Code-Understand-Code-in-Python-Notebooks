#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   svm-auc.py
#        \author   chenghuige  
#          \date   2014-01-04 08:58:40.965360
#   \Description   instatnce true probablity
# ==============================================================================

import sys,os
from sklearn.metrics import roc_auc_score

lines = open(sys.argv[1]).readlines()[1:]
line_list = [line.strip().split() for line in lines]
labels = [int(float((l[1]))) for l in line_list]

predicts = [float(l[4]) for l in line_list] 
score = roc_auc_score(labels, predicts)

print "auc: %f"%score
