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

if len(sys.argv) < 2:
  idir = sys.argv[1]
  files = glob.glob(f'{idir}/valid_*.csv')
else:
  files = sys.argv[1:]
  idir = './'

dfs = [pd.read_csv(file_) for file_ in files]
for i in range(len(dfs)):
  if i > 0:
    assert len(dfs[i]) == len(dfs[i-1])
df = pd.concat(dfs)

df = df.groupby('id', as_index=False).max()
df = df.sort_values('id')

df = df.to_csv(f'{idir}/ensemble.csv', index=False)
  
