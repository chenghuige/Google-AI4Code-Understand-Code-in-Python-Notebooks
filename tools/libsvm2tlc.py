#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   libsvm2tlc.py
#        \author   chenghuige  
#          \date   2014-01-09 11:26:02.819580
#   \Description  
# ==============================================================================

import sys,os

out_file = ""

if (len(sys.argv) > 3):
	out_file = sys.argv[3]
else:
	out_file = sys.argv[2].replace('.txt','.scaled.txt')
out = open(out_file, 'w')

header = open(sys.argv[2]).readline().strip()

out.write("%s\n"%header)

feature_num = len(header.split())

lines = open(sys.argv[1]).readlines()

for line in lines:
	l = line.strip().split()
	label = l[0]
	out.write(l[0])
	d = {}
	for item in l[1:]:
		(index, value) = item.split(':')
		d[int(index)] = value
	for index in range(feature_num):
		value = '0'
		if (index in d):
			value = d[index]
		out.write('\t' + value)
	out.write('\n')
		
