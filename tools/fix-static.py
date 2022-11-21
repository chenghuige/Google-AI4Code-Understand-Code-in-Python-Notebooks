#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   fix-static.py
#        \author   chenghuige  
#          \date   2014-04-17 11:17:33.911063
#   \Description  
# ==============================================================================

import sys,os

namespaces = [];
lsum = 0
rsum = 0
classes = [];
pre_isclass = True
pre_bracket = False 
input = sys.stdin
if len(sys.argv) > 1:
    input = open(sys.argv[1])
for line in input:
	line = line.strip()
	if (not pre_isclass and line.startswith('{')):
		pre_bracket = True 
	elif (pre_bracket and line.startswith('}')): #only ignore static variable in static functions 
		pre_bracket = False
	
	if line.startswith('namespace'):
		namespaces.append(line.split()[1])
	else:
		lsum += line.count('{')
		rsum += line.count('}')
                if rsum == lsum and rsum > 0:
                    if len(classes) > 0:
                        classes.pop()
	#print line,lsum,rsum
	#if ((line.startswith('struct') or line.startswith('class') or line.startswith('enum')) and not line.endswith(';')):
	if ((line.startswith('struct') or line.startswith('class') or line.startswith('enum'))):
		pre_isclass = True
		classes.append(line.split()[1])
	else:
		pre_isclass = False
	
	if (line.startswith('#') or line.startswith('/') or line.endswith(')') or line.replace(' ', '').endswith(');')or pre_bracket):
		continue
	if line.find('static const') >= 0 or line.find('const static') >= 0 and lsum > rsum:
		idx = line.rfind('=')
		if idx >= 0:
			line = line[:idx]

		idx = line.find('static const')
		if idx < 0:
			idx = line.find('const static')
		line = line[idx + len('static const'):]
		l = line.split()
		static_def_ = 'const ' + l[0] + ' ' + '::'.join(namespaces + classes) + '::' + l[1] + ';'
		if (len(classes) > 0):
			print static_def_
		else:
			#print static_def_ + '//global'
			pass
	
	


 
