#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cuda-version.py
#        \author   chenghuige  
#          \date   2019-07-21 14:04:38.111030
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

os.system('nvidia-smi')
os.system('nvcc --version')

try:
    head_file = open('/usr/local/cuda/include/cudnn.h')
except:
    head_file = open('/usr/include/cudnn.h')
lines = head_file.readlines()
for line in lines:
    line = line.strip()
    if line.startswith('#define CUDNN_MAJOR'):
        line = line.split('#define CUDNN_MAJOR')
        n1 = int(line[1])
        continue
    if line.startswith('#define CUDNN_MINOR'):
        line = line.split('#define CUDNN_MINOR')
        n2 = int(line[1])
        continue
    if line.startswith('#define CUDNN_PATCHLEVEL'):
        line = line.split('#define CUDNN_PATCHLEVEL')
        n3 = int(line[1])
        break
print('CUDNN Version ', str(n1)+'.'+str(n2)+'.'+str(n3))
