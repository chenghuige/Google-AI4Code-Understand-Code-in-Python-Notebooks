#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   to-simplify.py
#        \author   chenghuige  
#          \date   2018-10-22 07:35:13.154660
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import six 
assert six.PY2 

import gezi 

for line in sys.stdin:
  print(gezi.to_simplify(line))
