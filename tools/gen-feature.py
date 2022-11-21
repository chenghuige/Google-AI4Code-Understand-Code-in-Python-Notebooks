#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-feature.py
#        \author   chenghuige  
#          \date   2016-10-30 15:32:30.686419
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

for line in sys.stdin:
  l = line.rstrip().split()
  print(','.join(l))
