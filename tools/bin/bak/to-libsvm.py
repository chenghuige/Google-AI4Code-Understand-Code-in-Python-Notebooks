#!/usr/bin/env python

import sys,os
for line in open(sys.argv[1]):
    l = line.strip().split('\t')
    l2 = []
    l2.append(l[0])
    for i in range(3, len(l)):
        idx = i - 2
        val = l[i]
        result = "%d:%s"%(idx, val)
        l2.append(result.strip())
    print ' '.join(l2)
