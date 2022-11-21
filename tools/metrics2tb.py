#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   metrics2tb.py
#        \author   chenghuige  
#          \date   2020-04-20 00:58:00.005064
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import gezi
import pandas as pd

idir = sys.argv[1]
os.system(f'rm -rf {idir}/events*')
  
writer = gezi.SummaryWriter(idir, False)

df = pd.read_csv(f'{idir}/metrics.csv')

steps = df.step.values

keys =[x for x in df.columns if x != 'step']
for key in keys:
  for val, step in zip(df[key].values, steps):
    writer.scalar(key, val, step)


