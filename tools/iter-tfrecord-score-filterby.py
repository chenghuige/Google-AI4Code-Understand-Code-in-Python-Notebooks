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
import melt
import gezi
from tfrecord_lite import decode_example 
import tensorflow as tf

in_dir = sys.argv[2]
files = gezi.list_files(in_dir)
total = melt.get_num_records(files)

def get_item(files):
  for file in files:
    for it in tf.compat.v1.python_io.tf_record_iterator(file):
      yield it

if len(sys.argv) > 3:
  ofile = sys.argv[3]
  out = open(ofile, 'w') 
else:
  out = sys.stdout

if sys.argv[1] == 'none' or sys.argv[1] == 'all':
  ids = None
else:
  ids = set(sys.argv[1].split(','))

for it in tqdm(get_item(files), total=total):
  x = decode_example(it)
  if ids and str(x['abtestid'][0]) not in ids:
    continue
  print(x['mid'][0].decode(), x['docid'][0].decode(), x['duration'][0], x['ori_lr_score'][0], x['lr_score'][0], x['show_time'][0], sep='\t', file=out)

