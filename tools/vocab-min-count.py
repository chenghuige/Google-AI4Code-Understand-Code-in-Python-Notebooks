#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   vocab-min-count.py
#        \author   chenghuige  
#          \date   2018-04-29 17:31:02.253600
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

min_count = int(sys.argv[1])

for line in sys.stdin:
  word, count = line.rstrip('\n').split('\t')
  if int(count) < min_count:
    break
  print(word, count, sep='\t')

