#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   filter.py
#        \author   chenghuige  
#          \date   2019-10-24 09:02:02.709027
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

for line in sys.stdin:
  score = line.strip().split()[-1]
  #print(score)
  if len(score) >= len('0.xxxx88'):
    if score[6] == '8' and score[7] == '8':
      print(line.strip())

