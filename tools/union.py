#!/usr/bin/env python
#coding=gbk
import sys
file1 = sys.argv[1]
file2 = sys.argv[2]

a = set()

#id = 0
for line in open(file1):
    line = line.strip()
    #if (id % 2 == 0):
    a.add(line)
    #id += 1

#id = 0
for line in open(file2):
    line = line.strip()
    #if (id % 2 == 0):
    a.add(line)
    #id += 1
print len(a)