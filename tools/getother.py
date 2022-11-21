#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   rsync.py
#        \author   chenghuige  
#          \date   2013-10-24 11:19:28.376813
#   \Description  
# ==============================================================================

import sys,os
from subprocess import *

lprefix = '/home/gezi'
rprefix = '/home/users/chenghuige'

cmd = 'pwd'
path = Popen(cmd, shell = True, stdout = PIPE).stdout.readline().strip()

if(len(sys.argv) == 2):
    path += ('/' + sys.argv[1]) 
else:
    path += '/*'

if path.startswith(lprefix):
    path = path.replace(lprefix, rprefix)
elif path.startswith(rprefix):
    path = path.replace(rprefix, lprefix)

src = path

cmd = 'rsync -au {} .'.format(src)

print cmd
os.system(cmd)
