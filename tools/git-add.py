#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   git-add.py
#        \author   chenghuige  
#          \date   2015-06-27 22:38:53.449111
#   \Description  
# ==============================================================================

import sys,os
import gezi 

for f in gezi.get_filepaths(sys.argv[1]):
	if f.endswith('.h') or f.endswith('.cpp') or f.endswith('.cc') or f.endswith('.hpp') or f.endswith('.c') or f.endswith('.py') or f.endswith('.sh') or f.endswith('.hql') or f.endswith('.perl') or f.endswith('COMAKE') or f.endswith('Makefile'):
		cmd = 'git add ' + f 
		print cmd
		os.system(cmd)

 
