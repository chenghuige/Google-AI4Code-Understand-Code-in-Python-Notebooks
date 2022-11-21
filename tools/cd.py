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

if path.startswith(lprefix):
    path = path.replace(lprefix, rprefix)
elif path.startswith(rprefix):
    path = path.replace(rprefix, lprefix)


cmd = 'cd {}'.format(path)
print cmd
os.system(cmd)
