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

rhost = "cp01-rdqa-pool405.cp01.baidu.com"
cmd = 'pwd'
path = Popen(cmd, shell = True, stdout = PIPE).stdout.readline().strip()

src = '.'
if (src.endswith('/')):
	path += '/' + src
dest = '%s:%s'%(rhost,path)
cmd = 'ssh %s mkdir -p %s'%(rhost,path)
print cmd
os.system(cmd)
cmd = 'rsync -r -l -t -u %s %s'%(src,dest)
print cmd
os.system(cmd)

rhost = "cp01-rdqa-pool408.cp01.baidu.com"
cmd = 'pwd'
path = Popen(cmd, shell = True, stdout = PIPE).stdout.readline().strip()

src = '.'
if (src.endswith('/')):
	path += '/' + src
dest = '%s:%s'%(rhost,path)
cmd = 'ssh %s mkdir -p %s'%(rhost,path)
print cmd
os.system(cmd)
cmd = 'rsync -r -l -t -u %s %s'%(src,dest)
print cmd
os.system(cmd)

if len(sys.argv) > 2:
	cmd = "ssh %s 'cd %s;mpirun --machinefile /home/users/chenghuige/hosts -np %s %s'"%(rhost, path, sys.argv[2], sys.argv[1])
else:
	cmd = "ssh %s 'cd %s;mpirun --machinefile /home/users/chenghuige/hosts %s'"%(rhost, path, sys.argv[1])

print cmd
os.system(cmd)
cmd = 'rsync -r -l -t -u %s/* %s'%(dest,src)
print cmd
os.system(cmd)

