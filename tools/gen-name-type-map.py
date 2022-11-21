#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gen-name-type-map.py
#        \author   chenghuige  
#          \date   2014-11-15 21:41:57.971702
#   \Description  
# ==============================================================================

import sys,os

is_first = True
enum_name = ''
enum_map_name = ''
for line in sys.stdin:
	line = line.strip().strip(',').strip('{')
	if len(line) == 0 or line.startswith('//'):
		continue
	if is_first:
		is_first = False
		enum_name = line.split()[-1]
		enum_map_name = '_' + enum_name[0].lower() + enum_name[1:] + 's'
		print 'map<string, %s> %s = {'%(enum_name, enum_map_name)
	else:
		name = line
		print '{ \"%s\", %s },'%(name.lower().replace('-','').replace('_',''), enum_name + '::' + name)

print '};'
	

