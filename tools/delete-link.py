#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   delete-empty.py
#        \author   chenghuige  
#          \date   2020-02-27 09:08:23.383813
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

root = './'
if len(sys.argv) > 1:
  root = sys.argv[1]

files = os.listdir(root)
for file in files:
  if os.path.islink(file):  
    print('link:', file)
    os.system(f'rm -rf {file}')
  
