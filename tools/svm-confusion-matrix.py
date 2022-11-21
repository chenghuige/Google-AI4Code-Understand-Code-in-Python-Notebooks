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
if (len(sys.argv) > 3):
	thre = float(sys.argv[3])
print "thre: %f"%thre

labels = open(sys.argv[1]).readlines()
label_list = [int(float((line.strip().split()[0]))) for line in labels]

predicts = open(sys.argv[2]).readlines()[1:]
predicts = [float(line.strip().split()[1]) for line in predicts]
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

num_pos = tp + fn
num_neg = tn + fp 
pratio = num_pos / float(num_pos + num_neg)

tpr = tp * 1.0 / num_pos
tnr = tn * 1.0 / num_neg
print """
TEST POSITIVE RATIO:	%f (%d/(%d+%d))

Confusion table:
         ||===============================|
         ||            PREDICTED          |
  TRUTH  ||    positive    |   negative   | RECALL
         ||===============================|
 positive||   %d           |    %d        | 0.8529 (29/34)
 negative||   %d           |    %d        | 0.9538 (62/65)
         ||===============================|
 PRECISION 0.9063 (29/32)    0.9254(62/67)

OVERALL 0/1 ACCURACY:		0.9192 (91/99)
"""%(pratio, num_pos, num_pos, num_neg, tp, fn, fp, tn, )


print "For label 1: POS"
print "Precision: %f"%(tp * 1.0 / (tp + fp))
print "Recall: %f"%(tp * 1.0 / (tp + fn))

print "For label 0: NEG"
print "Precision: %f"%(tn * 1.0 / (tn + fn))
print "Recall: %f"%(tn * 1.0 / (tn + fp))

print "Total precision: %f"%((tp + tn) * 1.0 / (tp + tn + fp + fn))
