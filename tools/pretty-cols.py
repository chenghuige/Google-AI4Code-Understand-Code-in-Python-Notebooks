#!/usr/bin/env python
# ==============================================================================
#          \file   get-col.py
#        \author   chenghuige  
#          \date   2014-10-02 12:46:49.764521
#   \Description  
# ==============================================================================

import sys,os
from prettytable import PrettyTable  

pt = PrettyTable(encoding='utf8')

if len(sys.argv) < 2:
  indexes = []
else:
  indexes = [int(x) for x in sys.argv[1].split(',')]

def guess_sep(line):
  seps = ['\t', ' ', '\a', ',']
  words = 0
  sep_ = None
  for sep in seps:
    l = line.split(sep)
    if len(l) > words:
      sep_ = sep
      words = len(l)
  return sep_

for i, line in enumerate(sys.stdin):
  line = line.strip()
  if i == 0:
    sep = guess_sep(line)
  l = line.split(sep)
  if not indexes:
    indexes = range(len(l))
  #print '\t'.join(l[x] for x in indexes)
  pt.add_row([l[x] for x in indexes])

pt.align = 'l'
print(pt)
