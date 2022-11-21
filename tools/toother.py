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
lprefix2 = '/home/gezi2'
rprefix = '/home/users/chenghuige'

cmd = 'pwd'
path = Popen(cmd, shell = True, stdout = PIPE).stdout.readline().strip()

src = '*'

if(len(sys.argv) == 2):
	src = sys.argv[1]
        if (src.endswith('/')):
            path += '/' + src

if path.startswith(lprefix):
    path = path.replace(lprefix, rprefix)
elif path.startswith(lprefix2):
	path = path.replace(lprefix2, rprefix)
elif path.startswith(rprefix):
    path = path.replace(rprefix, lprefix)

dest = path

cmd = 'mkdir -p {}'.format(dest)
print cmd
os.system(cmd)

cmd = 'rsync -au {} {}'.format(src,dest)

print cmd
os.system(cmd)
