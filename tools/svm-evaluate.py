#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   svm-evaluate.py
#        \author   chenghuige  
#          \date   2014-01-04 08:58:40.965360
#   \Description   similar to evluate.py but need two input just for libsvm
#                  input1： featur(.libsvm) for label
#                  input2:  libsvm test output(.predict) for probability
# ==============================================================================

import sys,os
#---------------------------------thre
thre = 0.5
if (len(sys.argv) > 2):
	thre = float(sys.argv[2])
print "Thre: %.4f"%thre

#---------------------------------deal input
labels = open(sys.argv[1]).readlines()
label_list = [int(float((line.strip().split()[0]))) for line in labels]

predicts = open(sys.argv[2]).readlines()
header = predicts[0]
predicts = predicts[1:]
idx = 1
names = header.split()
for i in range(len(names)):
  if (names[i] == '1'):
	idx = i
predicts = [float(line.strip().split()[idx]) for line in predicts]
predict_list = [int(item > thre) for item in predicts]

#---------------------------------confusion table
tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(label_list)):
	if (predict_list[i] == 1):
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
num_neg = fp + tn
total_instance = num_pos + num_neg
pratio = num_pos * 1.0 / total_instance

#true positive rate
tpr = tp * 1.0 / num_pos
tnr = tn * 1.0 / num_neg

#num of predicted positive
num_pp = tp + fp
num_pn = fn + tn
#tur postive accuracy
tpa = tp * 1.0 / num_pp
tna = tn * 1.0 / num_pn

ok_num = tp + tn
accuracy = ok_num * 1.0 / total_instance

print """
TEST POSITIVE RATIO:	%.4f (%d/(%d+%d))

Confusion table:
         ||===============================|
         ||            PREDICTED          |
  TRUTH  ||    positive    |   negative   | RECALL
         ||===============================|
 positive||    %-5d       |   %-5d      | %.4f (%d / %d)
 negative||    %-5d       |   %-5d      | %.4f (%d / %d)
         ||===============================|
 PRECISION %.4f (%d/%d)    %.4f(%d/%d)

OVERALL 0/1 ACCURACY:		%.4f (%d/%d)
"""%(pratio, num_pos, num_pos, num_neg, tp, fn, tpr, tp, num_pos, fp, tn, tnr, tn, num_neg, tpa, tp, num_pp, tna, tn, num_pn, accuracy, ok_num, total_instance)

#----------------------------------------------------- auc area
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(label_list, predicts)

print """
ACCURACY:            %.4f
POS. PRECISION:      %.4f
POS. RECALL:         %.4f
NEG. PRECISION:      %.4f
NEG. RECALL:         %.4f
AUC:                 %.4f
"""%(accuracy, tpa, tpr, tna, tnr, auc)


#------------------------------------------------------roc curve
import pylab as pl
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(label_list, predicts)
roc_auc = auc(fpr, tpr)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
#pl.show()
pl.savefig(sys.argv[1] + '.roc.jpg')

