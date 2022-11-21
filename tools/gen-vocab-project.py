#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-vocab-project.py
#        \author   chenghuige  
#          \date   2016-12-14 19:16:44.724458
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
#by defualt will <PAD0> not in vocab.txt 
num_reserved = 1
if len(sys.argv) > 1:
  num_reserved = int(sys.argv[1])

for i in xrange(num_reserved):
  print('<PAD{}>'.format(i))

for line in sys.stdin:
  print(line.split('\t')[0])
