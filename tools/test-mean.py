#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pd-mean.py
#        \author   chenghuige  
#          \date   2020-04-21 14:02:19.183349
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import glob 

idir = sys.argv[1]

if os.path.isdir(idir):
  files = glob.glob(f'{idir}/test_*.csv')
else:
  idir = './'
  files = sys.argv[1:]

dfs = [pd.read_csv(file_) for file_ in files]
print(len(dfs))
for i in range(len(dfs)):
  if i > 0:
    assert len(dfs[i]) == len(dfs[i-1])

df = pd.concat(dfs)

key = 'id' if 'id' in df.columns else 'index'
df = df.groupby(key, as_index=False).mean()
df = df.sort_values(key)

df = df.to_csv(f'{idir}/submission.csv', index=False)
  
