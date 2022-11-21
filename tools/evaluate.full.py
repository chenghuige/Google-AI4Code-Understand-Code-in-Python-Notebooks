#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2014-01-04 08:58:40.965360
#   \Description   similary to TLC show : confusion matrix , pic of auc
#                  input is one file : instance,true,probability,assigned,..
#                  for libsvm test, need to file as input feature(.libsvm) and result(.predict) ->svm-evluate.py or svm-gen-evaluate.py first
#                  for tlc the header format is: instance,true, assigned,output, probability 
#                  TODO understancd other output of tlc and add more
# ==============================================================================

import sys
import os
import glob
from gflags import *
import matplotlib
matplotlib.use("agg")
#import pylab as pl
import matplotlib.pylab as pl 

#hack for some machine sklearn/externals/joblib/parallel.py:41: UserWarning: This platform lacks a functioning sem_open implementation, therefore, the required synchronization primitives needed will not function, see issue 3770..  joblib will operate in serial mode
import warnings
warnings.filterwarnings("ignore") 
#hack for cxfreeze
import sklearn.utils.sparsetools._graph_validation
from scipy.sparse.csgraph import _validation
from sklearn.utils import lgamma

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

DEFINE_boolean('show', True, 'wehter to show the roc pic')
DEFINE_float('thre', 0.5, 'thre for desciding predict')
DEFINE_string('image', 'temp.roc.pr.png', 'output image')
DEFINE_integer('max_num', 20, 'most to deal')
DEFINE_string('regex', '', 'use regex to find files to deal')
DEFINE_string('column', 'probability', 'score index name')
DEFINE_float('roc_xlim', 1.0, '')

from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
  
#---------------------------------------------------  
  
xmajorLocator   = MultipleLocator(0.1) 
xmajorFormatter = FormatStrFormatter('%1.1f') 
xminorLocator   = MultipleLocator(0.01) 
   
ymajorLocator   = MultipleLocator(0.02) 
ymajorFormatter = FormatStrFormatter('%1.3f')
yminorLocator   = MultipleLocator(0.01)

_width = 20
_height = 10
_dpi = 1
#confusion matrix, auc, roc curve
def evaluate(label_list, predicts, predict_list, file_name):
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
    tpr = 1
    tnr = 1
    if num_pos != 0:
      tpr = tp * 1.0 / num_pos
    if num_neg != 0:
      tnr = tn * 1.0 / num_neg
    
    #num of predicted positive
    num_pp = tp + fp
    num_pn = fn + tn
    #tur postive accuracy
    tpa = 1
    tna = 1
    if num_pp != 0:
        tpa = tp * 1.0 / num_pp
    if num_pn != 0:
        tna = tn * 1.0 / num_pn
    
    ok_num = tp + tn
    accuracy = ok_num * 1.0 / total_instance
    
    print """
    TEST POSITIVE RATIO:    %.4f (%d/(%d+%d))
    
    Confusion table:
             ||===============================|
             ||            PREDICTED          |
      TRUTH  ||    positive    |   negative   | RECALL
             ||===============================|
     positive||    %-5d       |   %-5d      | [%.4f] (%d / %d)
     negative||    %-5d       |   %-5d      |  %.4f  (%d / %d) wushang:[%.4f]
             ||===============================|
     PRECISION [%.4f] (%d/%d)   %.4f(%d/%d)
    
    OVERALL 0/1 ACCURACY:        %.4f (%d/%d)
    """ % (pratio, num_pos, num_pos, num_neg, tp, fn, tpr, tp, num_pos, fp, tn, tnr, tn, num_neg, 1 - tnr, tpa, tp, num_pp, tna, tn, num_pn, accuracy, ok_num, total_instance)
    
    #----------------------------------------------------- auc area
    #auc = roc_auc_score(label_list, predicts)
    
    fpr_, tpr_, thresholds = roc_curve(label_list, predicts)
    roc_auc = auc(fpr_, tpr_)
    
    print """
    ACCURACY:            %.4f
    POS. PRECISION:      %.4f
    POS. RECALL:         %.4f
    NEG. PRECISION:      %.4f
    NEG. RECALL:         %.4f
    AUC:                [%.4f]
    """ % (accuracy, tpa, tpr, tna, tnr, roc_auc)
    
    #------------------------------------------------------roc curve
    #pl.clf()
    pl.plot(fpr_, tpr_, label='%s: (area = %0.4f)' % (file_name, roc_auc))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, FLAGS.roc_xlim])
    pl.ylim([0.0, 1.0])
    pl.grid(True) 
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Roc Curve:')
    pl.legend(loc="bottom right") 

def parse_input(input):
  lines = open(input).readlines()
  header = lines[0]
  lines = lines[1:]
  
  label_idx = 1
  output_idx = 3
  probability_idx = 4
  names = header.split()
  for i in range(len(names)):
    if (names[i].lower() == 'label' or names[i].lower() == 'true'):
      label_idx = i
    if (names[i].lower() == 'output'): 
    	output_idx = i
    if (names[i].lower() == FLAGS.column.lower()):
      probability_idx = i
  try:
    line_list = [line.strip().split('\t') for line in lines]
    label_list = [int(float((l[label_idx]))) for l in line_list]
  
    predicts = [float(l[probability_idx]) for l in line_list] 
    #predicts = [float(l[output_idx]) for l in line_list] 
    predict_list = [int(item >= FLAGS.thre) for item in predicts]
    return label_list, predicts, predict_list 
  except Exception:
    print "label_idx: " + str(label_idx) + " prob_idx: " + str(probability_idx)
    exit(1)
  
def precision_recall(label_list, predicts, file_name):
  # Compute Precision-Recall and plot curve
  precision, recall, thresholds = precision_recall_curve(label_list, predicts)
  area = auc(recall, precision)
  #print("Area Under Curve: %0.2f" % area)
  #pl.clf()
  pl.plot(recall, precision, label='%s (area = %0.4f)' % (file_name, area))
 
  pl.ylim([0.0, 1.0])
  pl.xlim([0.0, 1.0])
  pl.subplot(121).xaxis.set_major_locator(xmajorLocator)  
  pl.subplot(121).xaxis.set_major_formatter(xmajorFormatter)  
  
  pl.subplot(121).yaxis.set_major_locator(ymajorLocator)  
  pl.subplot(121).yaxis.set_major_formatter(ymajorFormatter)  
  
  pl.subplot(121).xaxis.grid(True, which='major')  
  pl.subplot(121).yaxis.grid(True, which='major') 
  #pl.grid(True)
  pl.xlabel('Recall (TPR)')
  pl.ylabel('Precision (Positive pridictive value)')
  pl.title('Precision-Recall curve')
  pl.legend(loc="lower left")

def main(argv):
  try:
    argv = FLAGS(argv)  # parse flags
  except gflags.FlagsError, e:
    print '%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS)
    sys.exit(1)
  
  pos = len(argv) - 1
  try:
    FLAGS.thre = float(argv[-1])
    pos -= 1
  except Exception:
    pass
  #---------------------------------thre
  print "Thre: %.4f" % FLAGS.thre
  #---------------------------------deal input
  l = []
  if (FLAGS.regex != ""):
    print "regex: " + FLAGS.regex
    l = glob.glob(FLAGS.regex)
    print l
  else:
    input = argv[1]
    l = input.split()
  if (len(l) > 1):
    FLAGS.show = True
    if (len(l) > FLAGS.max_num):
      l = l[:FLAGS.max_num]
    #deal with more than 1 input
    f = pl.figure("Model Evaluation",figsize=(_width, _height), dpi = _dpi)
    f.add_subplot(1, 2, 0)
    for input in l:
      print "--------------- " + input
      label_list, predicts, predict_list = parse_input(input)
      evaluate(label_list, predicts, predict_list, input)
    f.add_subplot(1, 2, 1)
    for input in l:
      label_list, predicts, predict_list = parse_input(input)
      precision_recall(label_list, predicts, input)
  else:
    input2 = ""  
    if (pos > 1):
      input2 = argv[2]
      #FLAGS.show = True
    print "--------------- " + input
    label_list, predicts, predict_list = parse_input(input)
    f = pl.figure(figsize=(_width, _height))
    f.add_subplot(1, 2, 0)
    evaluate(label_list, predicts, predict_list, input)
    
    print "--------------- " + input2
    label_list2 = []
    predicts2 = []
    predict_list2 = []
    if (input2 != ""):
      label_list2, predicts2, predict_list2 = parse_input(input2)
      evaluate(label_list2, predicts2, predict_list2, input2)
  
    f.add_subplot(1, 2, 1)
    precision_recall(label_list, predicts, input)
  
    if (input2 != ""):  
      precision_recall(label_list2, predicts2, input2)

  pl.savefig(FLAGS.image)
  if (FLAGS.show):
    pl.show()
  
    
if __name__ == "__main__":  
  main(sys.argv)  
