#!/usr/bin/env python
# ==============================================================================
#          \file   pread-dir.py
#        \author   chenghuige  
#          \date   2017-09-02 22:06:24.379890
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import glob
from multiprocessing import Pool
  
exe = sys.argv[1]
idir = sys.argv[2].strip('/')
odir = sys.argv[3].strip('/')
os.system('mkdir -p %s'%odir)

def run(ifile):
  print('convert %s to dir %s'%(ifile, odir), file=sys.stderr)
  os.system('cat {} | python {} {}'.format(ifile, exe, odir))

files = glob.glob(idir + '/*')
pool = Pool()
pool.map(run, files)
pool.close()
pool.join()
