#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show-metrics.py
#        \author   chenghuige  
#          \date   2019-11-28 20:15:26.147059
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import gezi

keys = ['info', 'gold/auc', 'group/auc', 'group/click/time_auc']

def parse(input):
  l = input.split()
  l = [x.split(':') for x in l]
  m = dict(l)
  if not 'gold/auc' in m:
    m['gold/auc'] = (m['group/auc'] + m['group/click/auc']) / 2.0
  return m

lines = open(sys.argv[1]).readlines()
for line in reversed(lines):
  l = line.strip().split('\t')
  if len(l) < 4:
    continue
  if l[1].endswith(sys.argv[2]) and l[0].startswith('valid_offline_abid4,5,6') and l[2] == 'video':
    m = parse(l[-1])
    m['info'] = l[1]
    gezi.pprint_dict(m, keys)
    break
  
  
