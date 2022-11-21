#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   svm-auc.py
#        \author   chenghuige  
#          \date   2014-01-04 08:58:40.965360
#   \Description   libsvm file  &  predict file
# ==============================================================================

import sys,os
from sklearn.metrics import roc_auc_score

labels = open(sys.argv[1]).readlines()
labels = [int(float((line.strip().split()[0]))) for line in labels]

predicts = open(sys.argv[2]).readlines()[1:]
predicts = [float(line.strip().split()[1]) for line in predicts] 
score = roc_auc_score(labels, predicts)

print "auc: %f"%score