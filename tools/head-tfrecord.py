#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-tfrecord.py
#        \author   chenghuige  
#          \date   2019-09-11 11:00:01.818073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import app, flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

from tfrecord_lite import decode_example


def main(_):
  it = tf.compat.v1.python_io.tf_record_iterator(sys.argv[1])

  count = 1
  if len(sys.argv) > 2:
    count = int(sys.argv[2])

  for i in range(count):
    print('-----------i', i)
    x = decode_example(next(it))
    print(x)
    break

if __name__ == '__main__':
  app.run(main)  
  
