#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   add-fake-rank-label.py
#        \author   chenghuige  
#          \date   2017-11-30 23:39:08.987390
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

for i, line in enumerate(sys.stdin):
  l = line.strip().split('\t')
  if i == 0:
    l.insert(2, 'label')
  else:
    l.insert(2, '4')
  print('\t'.join(l))
  
