#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   fix-enum-class-step2.py
#        \author   chenghuige  
#          \date   2015-06-22 18:51:46.351022
#   \Description  
# ==============================================================================

import sys,os
import re 

input = sys.stdin
if len(sys.argv) > 1:
    input = open(sys.argv[1])

patterns = []
class_names = []
for line in open('enum_class.txt'):
    class_name = line.strip()
    class_names.append(class_name)
    pattern = class_name + '\s+?[a-zA-Z0-9_]+\s*?=\s*?[a-zA-Z0-9_:]*?(' + class_name + '::[a-zA-Z0-9_]+)\s*[,|\)]'
    #print pattern
    patterns.append(re.compile(pattern))

for line in input:
    line = line.strip()
    if line.endswith(');'):
        find = False
        for i in range(len(patterns)):
            pattern = patterns[i]
            class_name = class_names[i]
            m = pattern.search(line)
            if m:
                #print m.group(1)
                name = m.group(1).split('::')[1]
                line = line.replace(m.group(1), class_name + '__enum__' + name)
                print line
                find = True
                break
        if find:
            continue
    print line
                




