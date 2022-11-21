#!/usr/bin/env  python
import random, os, sys
from sets import Set
f = open(sys.argv[1], 'rb')
w = open(sys.argv[2], 'w')

sample_num = int(sys.argv[3])
query_num  = 0

def calc(file_name):
    count = 0
    for line in open(file_name, 'rb'):
        count += 1
    return count

if (len(sys.argv) > 4):
    query_num = int(sys.argv[4])
else:
    #query_num  = int(os.popen('wc -l %s'%sys.argv[1]).read().split()[0])
    query_num = calc(sys.argv[1])
print query_num
l = xrange(query_num)
u = random.sample(l, sample_num)
u = Set(u)

num = 0
col = -1

if (len(sys.argv) > 5):
    col = int(sys.argv[5])

print "col: %d"%(col)
    
for line in f:
    if (num in u):
        if (col == -1):
            w.write("%s"%line)
			#print line,
        else:
            l = line.strip().split('\t')[col]
            w.write("%s\n"%l)
			#print line
    num += 1
    if (num % 100000 == 0):
        print num
                
