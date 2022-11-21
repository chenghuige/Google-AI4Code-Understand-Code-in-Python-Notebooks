#!/usr/bin/env python
# ==============================================================================
#          \file   convert-dir-utf82gbk.py
#        \author   chenghuige  
#          \date   2017-08-30 16:56:05.008755
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
  ofile = ifile.replace(idir, odir)
  if os.path.exists(ofile):
    print('exist %s continue'%ofile, file=sys.stderr)
    continue
  print('convert %s to %s'%(ifile, ofile), file=sys.stderr)
  os.system('cat {} | python {} > {}'.format(ifile, exe, ofile))
  num += 1
  if num_files and num >= num_files:
    break
