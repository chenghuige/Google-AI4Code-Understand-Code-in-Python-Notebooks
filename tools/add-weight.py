#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   add-weight.py
#        \author   chenghuige  
#          \date   2014-01-13 14:16:11.054583
#   \Description  
# ==============================================================================

import sys,os
from pandas import *

df = read_table(sys.argv[1])
df['weight'] = 1

condition = df['label'] == 0
df.loc[condition, 'weight'] = float(sys.argv[2])

names = ['Instance','Label','True','Assigned']
for name in names:
  try:
    df[name] = df[name].astype(int)
  except Exception:
    pass
  name = name.lower()
  try:
    df[name] = df[name].astype(int)
  except Exception:
    pass
ncols = len(df.loc[0])
print 'weight:%d'%(ncols - 1)
df.to_csv(sys.argv[1].replace('.txt', '.weight.txt'), index = False, sep = '\t')