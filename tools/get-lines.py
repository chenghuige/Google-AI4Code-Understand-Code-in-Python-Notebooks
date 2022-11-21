#!/usr/bin/env python
import sys
file = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
i = 0
for line in open(file):
  if (i >= end):
    break
  elif (i >= start):
    print line,
  i += 1
    
