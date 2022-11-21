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


rhost = "root@cq01-forum-urate04.cq01.baidu.com"

cmd = 'pwd'
path = Popen(cmd, shell = True, stdout = PIPE).stdout.readline().strip()

src = '.'

if(len(sys.argv) == 2):
	src = sys.argv[1]
        if (src.endswith('/')):
            path += '/' + src
dest = '%s:%s'%(rhost,path)

cmd = 'ssh %s mkdir -p %s'%(rhost,path)
print cmd

os.system(cmd)
cmd = 'rsync -r -l -t %s %s'%(src,dest)

print cmd
os.system(cmd)
