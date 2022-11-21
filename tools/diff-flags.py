#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   diff-flags.py
#        \author   chenghuige  
#          \date   2019-11-25 22:38:42.361655
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

a = {}
b = {}

def deal(file, m):
  for line in open(file):
    l = line.strip().split('=')
    if len(l) == 1:
      key = l[0]
      val = None
    elif len(l) == 2:
      key = l[0]
      val = l[1]
    else:
      continue
    m[key] = val 

deal(sys.argv[1], a)

deal(sys.argv[2], b)

seta = set(a.keys())
setb = set(b.keys())

l = setb.difference(seta)
print("b.difference a %d" % (len(l)))
for item in l:
  print(item)

l = seta.difference(setb)
print("a.difference b %d" % (len(l)))
for item in l:
  print(item)

for key in a:
  if key in b:
    if a[key] != b[key]:
      print(key, a[key], b[key])
