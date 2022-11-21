#!/usr/bin/env python
#coding=gbk
import sys
file1 = sys.argv[1]
file2 = sys.argv[2]

at = set()
bt = set()

at_map = {}
bt_map = {}

for line in open(file1):
    line = line.strip()
    l = line.split('\t')
    at_map[l[0]] = line
    at.add(l[0])

for line in open(file2):
    line = line.strip()
    l = line.split('\t')
    bt_map[l[0]] = line
    bt.add(l[0]) 


l = bt.difference(at)
print "new%d"%(len(l))
for item in l:
    print bt_map[item]

l = at.difference(bt)
print "old%d"%(len(l))
for item in l:
    print at_map[item]
