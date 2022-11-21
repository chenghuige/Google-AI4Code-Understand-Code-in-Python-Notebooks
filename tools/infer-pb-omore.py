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

in_dir = sys.argv[1]
files = gezi.list_files(in_dir)
random.shuffle(files)
num_records_file = os.path.join(in_dir, "num_records.txt")
total = melt.get_num_records(files) if not os.path.exists(num_records_file) else gezi.read_int_from(num_records_file)
print('total', total, file=sys.stderr)

def get_item(files):
  for file in files:
    for it in tf.compat.v1.python_io.tf_record_iterator(file):
      yield it

predictor = melt.Predictor()
pb_path = sys.argv[2]
print('model.pb:', pb_path, file=sys.stderr)
predictor.load_graph(pb_path, 'prefix', frozen_map_file=pb_path.replace('.pb', '.map'))

for it in tqdm(get_item(files), total=total):
  x = decode_example(it)
  if x['abtestid'] != 15:
    continue
  duration = x['duration'][0]
  if duration > 60 * 60 * 12:
    duration = 60
  id = x['id'][0].decode()
  mid, doc_id = id.split('\t')
  index, value, field = x['index'], x['value'], x['field']
  feed_dict = {predictor.graph.get_collection('index_feed')[0]: [index],
               predictor.graph.get_collection('value_feed')[0]: [value],
               predictor.graph.get_collection('field_feed')[0]: [field]}
  pred = predictor.predict('pred', feed_dict)
  pred = pred[0][0]
  print(mid, doc_id, duration, pred, sep='\t')
