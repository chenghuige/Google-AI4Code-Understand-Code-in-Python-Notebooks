#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-feature.py
#        \author   chenghuige  
#          \date   2016-10-30 15:32:30.686419
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

features = sys.argv[1].split(',')
fnames_file = './feature_name.txt'
if len(sys.argv) > 2:
  fnames_file = sys.argv[2]

fnames = []
for line in open(fnames_file):
  fnames.append(line.rstrip().rsplit('#')[0].strip())

for fname, feature in zip(fnames, features):
  print(fname, feature)
