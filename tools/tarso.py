#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   tarso.py
#        \author   chenghuige  
#          \date   2015-06-19 20:21:57.848456
#   \Description  
# ==============================================================================

import sys,os

content = os.popen('ldd %s'%sys.argv[1]).read().strip().split('\n')

libs = [item.split()[-2] for item in content]

len_ = len(libs)

libs = [item for item in libs if item.find('so') != -1]

if len(libs) < len_:
    print 'Some lib so not find, try ldd ',sys.argv[1]

os.system('rm -rf ./sos;mkdir -p ./sos')
commands = ['cp %s ./sos'%command for command in libs]

print commands
for command in commands:
    os.system(command)

os.system('rm -rf sos.tar.gz;cd sos;tar -zvcf ../sos.tar.gz *;cd ..')
 
