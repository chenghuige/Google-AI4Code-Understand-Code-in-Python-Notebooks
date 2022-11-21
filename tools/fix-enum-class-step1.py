#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   fix-enum-class-step1.py
#        \author   chenghuige  
#          \date   2015-06-22 18:51:38.019860
#   \Description  
# ==============================================================================

import sys,os

input = sys.stdin
if len(sys.argv) > 1:
    input = open(sys.argv[1])

#only support 2 formats
#enum class color
#{
#    green = 0,
#    black,
#};
#enum class color
#{
#    green = 0,
#    black
#};
#enum class color
#{
#    green,
#    black,
#};
#enum class color
#{
#    green,
#    black
#};
#enum class color {green, black};
#enum class color {green, black,};

out = open('enum_class.txt', 'a')
pre_enum_class = False
class_name = ''
for line in input:
    line = line.strip() 
    if line.startswith('enum class'):
        l = line.split()
        class_name = l[2]
        out.write(class_name + '\n');
        if line.endswith('};'):
            line = line[line.find('{') + 1 : line.find('};')]
            member_names = [class_name + '__enum__' + (item + ' ')[:item.find('=')].strip() for item in line.split(',') if item != '']
            print 'enum ' + class_name 
            print '{'
            for member in member_names:
                print member + ','
            print '};'
        else:
            pre_enum_class = True
            print line.replace('class', '')
        continue
    
    if pre_enum_class:
        if line.endswith('};'):
            pre_enum_class = False 
        else:
            if line.endswith(','):
                print (class_name + '__enum__' + line[:line.find('=')]).strip() + ','
                continue
            elif not (line.startswith('{') or line == ''):
                print (class_name + '__enum__' + (line + ' ')[:line.find('=')]).strip() + ','
                continue
        print line 
        continue
    
    print line 

 
