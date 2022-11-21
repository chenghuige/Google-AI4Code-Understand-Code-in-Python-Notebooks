#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   cross-validation.py
#        \author   chenghuige  
#          \date   2014-09-24 08:30:52.611936
#   \Description  
# ==============================================================================

import sys,os

train_prefix = "feature.train_";
test_prefix = "feature.test_"

cross_dir = sys.argv[1] + "/"
nr = int(sys.argv[2])
train_exe = sys.argv[3]
#test_exe = sys.argv[4]

train_prefix = cross_dir + train_prefix
test_prefix = cross_dir + test_prefix
#model_prefix = cross_dir + "model"
result_prefix = cross_dir + "temp.result."

for i in range(nr):
    train_file = train_prefix + str(i)
    test_file = test_prefix + str(i)
    #model_file = model_prefix + str(i)
    result_file = result_prefix + str(i)
    #train_cmd = train_exe + " " + train_file + " " + model_file
    train_cmd = train_exe + " " + train_file + " " + test_file + " " + result_file
    print train_cmd
    os.system(train_cmd)
    #test_cmd = test_exe + " " + test_file + " " + model_file + " " + result_file
    #print test_cmd
    #os.system(test_cmd)

suffix = 'txt'
if len(sys.argv) > 4:
    suffix = sys.argv[4]
out_file = cross_dir + "result." + suffix
out = open(out_file, 'w')

#assume each result file to have header!
for i in range(nr):
    result_file = result_prefix + str(i)
    lines = open(result_file).readlines()
    if i != 0:
        lines = lines[1:]
    out.writelines(lines)

os.system('~/tools/evaluate.py %s'%out_file)
