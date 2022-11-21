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
 
idir = sys.argv[1].strip('/')
odir = sys.argv[2].strip('/')

os.system('mkdir -p %s'%odir)
for ifile in glob.glob(idir + '/*'):
  ofile = ifile.replace(idir, odir)
  print('convert %s to %s'%(ifile, ofile), file=sys.stderr)
  os.system('cat {} | python ~/tools/utf82gbk.py > {}'.format(ifile, ofile))
