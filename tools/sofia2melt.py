#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   sofia2melt.py
#        \author   chenghuige  
#          \date   2014-10-06 17:25:52.614587
#   \Description  
# ==============================================================================
import sys,os
print "ModelName=Linear-Sofia-RocSVM"
l = open(sys.argv[1]).readline().split()
bias = l[0]
l = l[1:]
print "FeatureNum=%d"%len(l)
print "-1\tbias\t%s"%bias
for index, value in enumerate(l):
	if value != '0':
		print "{0}\tf{0}\t{1}".format(index, value)

