#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   svm-auc.py
#        \author   chenghuige  
#          \date   2014-01-04 08:58:40.965360
#   \Description   instatnce true probablity
# ==============================================================================

import sys,os

thre = 0.5
if (len(sys.argv) > 2):
	thre = float(sys.argv[2])
print "thre: %f"%thre

lines = open(sys.argv[1]).readlines()[1:]
line_list = [line.strip().split() for line in lines]
label_list = [int(float((l[1]))) for l in line_list]

predicts = [float(l[2]) for l in line_list] 

predicate_list = [int(item > thre) for item in predicts]

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(label_list)):
	if (predicate_list[i] == 1):
		if (label_list[i] == 1):
			tp += 1
		else:
			fp += 1
	else:
		if (label_list[i] == 1):
			fn += 1
		else:
			tn += 1

print 'predicted:'
print 'true  false'
print '%s    %s'%(tp, fn)
print '%s    %s'%(fp, tn)

print "For label 1: POS"
print "Precision: %f"%(tp * 1.0 / (tp + fp))
print "Recall: %f"%(tp * 1.0 / (tp + fn))

print "For label 0: NEG"
print "Precision: %f"%(tn * 1.0 / (tn + fn))
print "Recall: %f"%(tn * 1.0 / (tn + fp))

print "Total precision: %f"%((tp + tn) * 1.0 / (tp + tn + fp + fn))
