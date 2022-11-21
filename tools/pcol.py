#!/usr/bin/python
import sys
file = sys.argv[1]
col = int(sys.argv[2])

for line in open(file):
  line = line.strip().split()[col]
  print "%s"%line
    
