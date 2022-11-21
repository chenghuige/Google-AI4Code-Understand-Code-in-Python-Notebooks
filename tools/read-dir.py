#!/usr/bin/env python
# ==============================================================================
#          \file   read-dir.py
#        \author   chenghuige  
#          \date   2017-09-02 22:06:19.048990
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import glob

exe = sys.argv[1]
idir = sys.argv[2].strip('/')
odir = sys.argv[3].strip('/')
try:
  num_files = int(sys.argv[4])
except Exception:
  num_files = 0

os.system('mkdir -p %s'%odir)

num = 0
for ifile in glob.glob(idir + '/*'):
  print('convert %s to dir %s'%(ifile, odir), file=sys.stderr)
  os.system('cat {} | python {} {}'.format(ifile, exe, odir))
  num += 1
  if num_files and num >= num_files:
    break
  
