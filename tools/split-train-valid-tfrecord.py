#!/usr/bin/env python
# ==============================================================================
#          \file   split-train-valid.py
#        \author   chenghuige  
#          \date   2017-08-23 14:19:07.162602
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, os

infile = sys.argv[1]

train_file = infile.replace('.txt', '.train.txt') if infile.endswith('.txt') else infile + '.train'
valid_file = infile.replace('.txt', '.valid.txt') if infile.endswith('.txt') else infile + '.valid'

out_train = open(train_file, 'w')
out_valid = open(valid_file, 'w')

num_lines = 0
for line in open(infile):
  num_lines += 1

ratio = float(sys.argv[2])
valid_lines = int(num_lines * ratio) 

if float(valid_lines) < 1:
  valid_lines = int(num_lines * float(valid_lines))
else:
  valid_lines = int(valid_lines)

train_lines = num_lines - valid_lines

for idx, line in enumerate(open(infile)):
  if idx < train_lines:
    out_train.write(line)
  else:
    out_valid.write(line)
