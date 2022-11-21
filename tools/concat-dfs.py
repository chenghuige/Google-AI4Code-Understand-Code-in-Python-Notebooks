#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   concat-dfs.py
#        \author   chenghuige  
#          \date   2020-02-02 15:06:55.870156
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import glob 
import pymp
from tqdm import tqdm
from multiprocessing import cpu_count, Manager

dir = sys.argv[1]
iname = sys.argv[2]
ofile = sys.argv[3]

files = glob.glob(f'{dir}/{iname}*')

ps = min(cpu_count(), len(files))

dfs = Manager().list()
with pymp.Parallel(ps) as p:
  for i in tqdm(p.range(len(files)), ascii=True, desc='concat-dfs'):
    file = files[i]
    df = pd.read_csv(file)
    dfs += [df]
df = pd.concat(dfs, sort=False)

print('len(df)', len(df))
df.to_csv(f'{dir}/{ofile}', index=False)

  
