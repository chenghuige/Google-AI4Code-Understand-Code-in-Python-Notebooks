#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tfrecord2idx.py
#        \author   chenghuige  
#          \date   2019-08-20 07:41:24.941165
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
from tqdm import tqdm
from subprocess import call

idir = sys.argv[1]
odir = sys.argv[2]

records = gezi.list_files('%s/*' % idir)
os.system('mkdir -p %s' % odir)
for record in tqdm(records, ascii=True, desc='tfrecord2idx'):
  record_idx = os.path.join(odir, os.path.basename(record))
  #print(record, record_idx)
  call(['tfrecord2idx', record, record_idx])
  
