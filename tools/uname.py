#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   uname.py
#        \author   chenghuige  
#          \date   2015-03-01 12:22:17.825670
#   \Description  
# ==============================================================================

import sys,os

import nowarning
import libtieba

print libtieba.get_user_info(int(sys.argv[1])).userName
 
