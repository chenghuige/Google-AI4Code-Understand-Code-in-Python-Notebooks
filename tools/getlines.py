#!/usr/bin/env python
import sys
file = sys.argv[1]
start = int(sys.argv[2])
length = int(sys.argv[3])
outfile = sys.argv[4]
out = open(outfile, 'w')
i = 0
for line in open(file):
  if (i >= start + length):
    break
  elif (i >= start):
    out.write(line)
  i += 1
out.close()
  
    
