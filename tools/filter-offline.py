#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   filter.py
#        \author   chenghuige  
#          \date   2019-10-24 09:02:02.709027
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

keys = set()
for line in open(sys.argv[1]):
  l = line.rstrip().split()
  key = '%s\t%s' % (l[0], l[1])
  keys.add(key)

for line in sys.stdin:
  l = line.rstrip().split()
  key = '%s\t%s' % (l[0], l[1])
  if key not in keys:
    continue 
  print(line.strip())
  
