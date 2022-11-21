#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   toxic-diff.py
#        \author   chenghuige  
#          \date   2018-02-16 16:40:00.471648
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import pandas as pd

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

limit = 0 
if len(sys.argv) > 3:
  limit = int(sys.argv[3])

if limit:
  df1 = df1[:limit]
  df2 = df2[:limit]

print('add-------------')

ids1 = set(df1['id'])
ids2 = set(df2['id'])

num_diff = 0
for i, id_ in enumerate(df2['id']):
  if id_ not in ids1:
    print(i, id_, df2['comment'][i])
    num_diff += 1

print('num_diff is ', num_diff)

print('del-------------')
for i, id_ in enumerate(df1['id']):
  if id_ not in ids2:
    print(i, id_, df1['comment'][i])


