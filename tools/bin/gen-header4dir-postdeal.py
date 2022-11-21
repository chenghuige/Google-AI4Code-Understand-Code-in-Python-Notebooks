#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gen-header-all.py
#        \author   chenghuige  
#          \date   2014-04-17 07:42:37.239655
#   \Description  
# ==============================================================================

import sys,os

list_dirs = os.walk(sys.argv[1])
for root, dirs, files in list_dirs:
    for d in files:
        file_ = os.path.join(root, d)
        if file_.endswith('.h'):
            cmd = 'sh gen-header-postdeal.sh '+ file_  
            os.system(cmd)
