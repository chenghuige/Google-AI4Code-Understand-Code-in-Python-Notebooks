#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble-ln.py
#        \author   chenghuige  
#          \date   2017-10-29 15:08:55.615165
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import glob

pattern = sys.argv[1]
pattern = '%s*' % pattern 
#print('pattern', pattern)
l = glob.glob(pattern + '')
#print(l)

def get_name(path):
  model_name = os.path.basename(path)
  model_dir = os.path.dirname(path)
  #name = '_'.join([x for x in model_dir.split('/') if not x.startswith('.')] + [model_name])
  dir_name = os.path.dirname(model_dir)  # bypass epoch 
  dir_name1 = os.path.basename(dir_name)
  dir_name = os.path.dirname(dir_name) #bypass 0
  dir_name2 = os.path.basename(dir_name)
  name = '_'.join([dir_name2, dir_name1, model_name])
  print(model_name, model_dir, dir_name, name)
  return name

l2 = [get_name(x) for x in l]
print(l2)

for src, dest in zip(l, l2):
  command = 'ln -s %s %s' % (src, dest)
  print(command, file=sys.stderr)
  os.system(command)

