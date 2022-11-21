#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show-feature.py
#        \author   chenghuige  
#          \date   2019-10-21 16:17:28.518581
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

feat_file = sys.argv[1]
index_file = sys.argv[2]

mid = sys.argv[3]
docid = sys.argv[4]

names = ['None']
for i, line in enumerate(open(index_file)):
  key = line.strip()
  names.append(key)

for line in open(feat_file):
  l = line.strip().split('\t')
  mid_, docid_, feats = l[0], l[1], l[3]
  if mid_ == mid and docid_ == docid:
    feats = set([names[int(x)] for x in feats.split()])
    feats = sorted(list(feats))
    for feat in feats:
      print(feat)
    print('num features:', len(feats), file=sys.stderr)
    fields = set(x.split('\a')[0] for x in feats)
    fields = list(fields)
    fields.sort()
    for field in fields:
      print(field)
    print('num fields:', len(fields), file=sys.stderr)
    break
  

  
