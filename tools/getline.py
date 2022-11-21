#!/usr/bin/python
import sys
file = sys.argv[1]
num = int(sys.argv[2])
f = open(file)
i = 0
object =''
for line in f:
  if (i == num):
    print line.rstrip('\n')
    object = line.rstrip('\n')
    break
  i += 1
f.close()
if(len(sys.argv) > 3):
  outfile = sys.argv[3]
  out = open(outfile, 'w')
  out.write(object)
  out.write('\n')
  out.close()
  
    
