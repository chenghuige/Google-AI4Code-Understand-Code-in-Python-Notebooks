#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   get-all-uers.py
#        \author   chenghuige  
#          \date   2019-08-18 11:06:39.496266
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import collections
from collections import defaultdict
from tqdm import tqdm
import random
import melt
import gezi
from tfrecord_lite import decode_example 
import tensorflow as tf

predictor = melt.Predictor()
pb_path = sys.argv[1]
print('model.pb:', pb_path, file=sys.stderr)
predictor.load_graph(pb_path, 'prefix', frozen_map_file=pb_path.replace('.pb', '.map'))
print('load pb done', file=sys.stderr)

for line in sys.stdin:
  mid, docid, duration, indexes, fields, values = line.strip().split('\t')
  index = list(map(int, indexes.split()))
  value = list(map(float, values.split()))
  field = list(map(int, fields.split()))
  feed_dict = {predictor.graph.get_collection('index_feed')[0]: [index],
               predictor.graph.get_collection('value_feed')[0]: [value],
               predictor.graph.get_collection('field_feed')[0]: [field]}
  pred = predictor.predict('pred', feed_dict)
  pred = pred[0][0]
  print(mid, docid, duration, pred, sep='\t')
