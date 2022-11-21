#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   jup2py-postdeal.py
#        \author   chenghuige  
#          \date   2022-04-24 00:29:06.589427
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi
from gezi.common import * 

ifile = sys.argv[1]
if ifile.endswith('.ipynb'):
  ifile = ifile.replace('ipynb', 'py')
ic(ifile)
lines = open(ifile).readlines()

with open(ifile, 'w', encoding='utf8') as f:
  for line in lines:
    line = line.rstrip('\n')
    if line.startswith('get_ipython().system'):
      line = line.replace('get_ipython()', 'gezi')
    #if line and not ' ' in line and not '=' in line and not '(' in line: 
    #  line = f'ic({line})'
    print(line, file=f)

