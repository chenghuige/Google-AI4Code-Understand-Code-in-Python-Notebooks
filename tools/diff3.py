#!/usr/bin/env python
#coding=gbk
import sys
file1 = sys.argv[1]
file2 = sys.argv[2]

out1 = open(sys.argv[3], 'w')
out2 = open(sys.argv[4], 'w')

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

print len(at_map)
print len(bt_map)

l = bt.difference(at)
for item in l:
	out1.write(bt_map[item])
	out1.write('\n')

l = at.difference(bt)
for item in l:
	out2.write(at_map[item])
	out2.write('\n')
