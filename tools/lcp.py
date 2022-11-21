#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   lcp.py
#        \author   chenghuige  
#          \date   2018-05-23 13:06:34.938478
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob 

input = sys.argv[1]

base = os.path.basename(input)
os.system('mkdir -p %s' % base)

for item in glob.glob(input + '/*'):
  if os.path.isdir(item):
    print(item)
    folder = os.path.basename(item)
    command = 'mkdir -p %s/%s' %(base, folder)
    print(command)
    os.system(command)
    command = 'mkdir -p %s/%s/epoch' %(base, folder)
    os.system(command)
    os.system('rsync --progress -avz %s/event* %s/%s' % (item, base, folder))
    os.system('rsync --progress -avz %s/epoch/event* %s/%s/epoch' % (item, base, folder))
  
