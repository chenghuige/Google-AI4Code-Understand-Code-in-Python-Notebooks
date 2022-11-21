#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   diff-metrics.py
#        \author   chenghuige  
#          \date   2019-10-30 14:03:52.619851
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import pandas as pd
import gezi 

importants = ['info', 'click_ratio', 'time_per_user', 'time_per_click', \
	      'auc', 'time_auc', 'weighted_time_auc', 'click/time_auc', 'click/weighted_time_auc', \
              'group/auc', 'group/time_auc', 'group/weighted_time_auc', 'group/click/time_auc', 'group/click/weighted_time_auc']


def rename(key):
  return key.replace('weighted_time', 'wtime') \
            .replace('version', 'v') \
            .replace('group', 'g') \
            .replace('click', 'c')
 
infos = []
for line in sys.stdin:
  l = line.rstrip().split('\t')
  name = l[0]
  info = l[-1].split()
  info = dict(x.split(':') for x in info)
  info['info'] = name
  infos.append(info)
  
df = pd.DataFrame.from_dict(infos)
gezi.pprint_df(df, importants, rename_fn=rename)
  
