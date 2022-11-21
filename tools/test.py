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

path = '/home/users/chenghuige/'
cmd = 'cd {}'.format(path)
print cmd
os.system(cmd)
os.chdir(path)
