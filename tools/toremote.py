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

src_prefix = '/home/gezi'
dest_prefix = '/home/users/chenghuige'

cmd = 'pwd'
path = Popen(cmd, shell = True, stdout = PIPE).stdout.readline().strip()

src = '*'

if(len(sys.argv) == 2):
	src = sys.argv[1]
        if (src.endswith('/')):
            path += '/' + src

dest = path.replace(src_prefix, dest_prefix)
cmd = 'mkdir -p {}'.format(dest)
print cmd
os.system(cmd)

cmd = 'rsync -au {} {}'.format(src,dest)

print cmd
os.system(cmd)
