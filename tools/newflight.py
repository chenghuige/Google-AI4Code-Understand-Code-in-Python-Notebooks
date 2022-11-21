#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   newflight.py
#        \author   chenghuige  
#          \date   2022-11-11 09:14:55.063192
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

ifile = sys.argv[1]
src = sys.argv[2]
dest = sys.argv[3]
spattern = sys.argv[4]
dpattern = sys.argv[5]

lines = []
for line in open(ifile):
  line = line.strip()
  lines.append(line)
  if src in line:
    line = line.replace(src, dest)
    if spattern in line:
      line = line.replace(spattern, dpattern)

    lines.append(line)

print('\n'.join(lines))


