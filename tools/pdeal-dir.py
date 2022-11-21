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
from multiprocessing import Pool

exe = sys.argv[1]
idir = sys.argv[2].strip('/')

os.system('mkdir -p %s'%odir)

def run(ifile):
  ofile = ifile.replace(idir, odir)
  if os.path.exists(ofile):
    print('exist %s continue'%ofile, file=sys.stderr)
    return 
  #print('convert %s to %s'%(ifile, ofile), file=sys.stderr)
  command = 'cat {} | python {} > {}'.format(ifile, exe, ofile)
  print(command, file=sys.stderr)
  os.system(command) 

files = glob.glob(idir + '/*')
pool = Pool()
pool.map(run, files)
pool.close()
pool.join()
