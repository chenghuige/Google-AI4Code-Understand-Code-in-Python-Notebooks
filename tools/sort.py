#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   sort.py
#        \author   chenghuige  
#          \date   2014-04-25 18:12:38.716273
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
import sys,os
flags.DEFINE_string('sep', '\t', 'seprator')
flags.DEFINE_boolean('reverse', True, 'sort reverse?')
flags.DEFINE_boolean('multi', True, '')
flags.DEFINE_boolean('header', False, '')
flags.DEFINE_integer('num_lines', None, '')

try:
  sys.argv = FLAGS(sys.argv)  # parse flags
except Exception:
  sys.exit(1)

col = int(sys.argv[1])
li = []
input = sys.stdin
if len(sys.argv) > 2:
  input = open(sys.argv[2])
for line in input:
  if FLAGS.header:
    FLAGS.header = False 
    print(line, end='')
  else:
    l = line.rstrip().split(FLAGS.sep)
    li.append((float(l[col]), line.rstrip()))
    
# print(li)
# exit(0)

li.sort(reverse = FLAGS.reverse)

if FLAGS.multi:
  num_lines = 0
  for item in li:
    print(item[1])
    num_lines += 1
    if FLAGS.num_lines and num_lines == FLAGS.num_lines:
      break
else:
  li2 = [item[1] for item in li]
  print('\t'.join(li2))

