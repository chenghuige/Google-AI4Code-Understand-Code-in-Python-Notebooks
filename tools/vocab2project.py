#!/usr/bin/env python
# ==============================================================================
#          \file   vocab2project.py
#        \author   chenghuige  
#          \date   2017-08-03 16:53:25.873308
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, os

import gezi 


vocab = gezi.Vocabulary(sys.argv[1])

ofile = sys.argv[1].replace('.txt', '.project')

with open(ofile, 'w') as out:
  print('Word\tFreqence', file=out)
  for i in range(vocab.size()):
    key = vocab.key(i)
    if not key.strip():
      key = 'empty%d' % i
    print(key, i, sep='\t', file=out)

