#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cp_model.py
#        \author   chenghuige  
#          \date   2019-10-30 00:20:59.308678
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import melt 
model = melt.latest_checkpoint(sys.argv[1])
command = 'rm -rf %s' % (model)
os.system(command)
  
