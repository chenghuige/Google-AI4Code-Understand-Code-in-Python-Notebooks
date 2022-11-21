#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   inspeckt-torch-model.py
#        \author   chenghuige  
#          \date   2020-03-01 17:18:40.877356
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import torch

input = sys.argv[1]

if os.path.isdir(input):
  import melt
  input = melt.latest_checkpoint(input, torch=True)

m = torch.load(input)
m = m['state_dict']

total = 0
for key in m.keys():
  total += sum(m[key].shape)
  print(key, m[key].shape)
print('total params:', total)
  
