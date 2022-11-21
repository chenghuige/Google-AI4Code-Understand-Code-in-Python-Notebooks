#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   svm-gen-evaluate.py
#        \author   chenghuige  
#          \date   2014-01-06 23:27:39.343801
#   \Description  input a output a.evaluate , internal use a.libsvm a.predict
# ==============================================================================

import sys,os

prefix = sys.argv[1]

label_file = prefix 
labels = open(label_file).readlines()
label_list = [line.strip().split()[0] for line in labels]

predict_file = prefix + ".predict"
predicts = open(predict_file).readlines()
header = predicts[0]
predicts = predicts[1:]
idx = 1
names = header.split()
for i in range(len(names)):
  if (names[i] == '1'):
	idx = i
predicts = [line.strip().split()[idx] for line in predicts]


outfile = prefix + ".evaluate"
out = open(outfile, 'w') 
out.write("true\tprobability\n")

for i in range(len(label_list)):
    out.write(label_list[i] + "\t" + predicts[i] + "\n")