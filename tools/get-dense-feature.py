#!/usr/bin/env python
import sys 
start = 2
if len(sys.argv) > 1:
	start = int(sys.argv[1])

for line in sys.stdin:
	l = line.strip().split()
	print l[:start]
	l = l[start:]
	l2 = ['%s:%s'%(i, item) for i,item in enumerate(l) if item != '0']
	print "'%s'"%(','.join(l2))
		
