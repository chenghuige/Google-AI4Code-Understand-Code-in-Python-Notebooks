#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   temp.py
#        \author   chenghuige  
#          \date   2019-10-19 00:30:38.453350
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

for line in sys.stdin:
  command = 'kill %s' % line.strip()
  print(command)  
  os.system(command)
