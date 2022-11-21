#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   utf8togbk.py
#        \author   chenghuige  
#          \date   2015-03-13 19:53:33.858613
#   \Description  
# ==============================================================================
from __future__ import print_function

import sys,os
import nowarning
import libgezi 

from libgezi import LogHelper
#for baidu lib, default 16 will print debug info, set 8 so not print debug info to much for convert 
logger = LogHelper(0)  

for line in sys.stdin:
	print(libgezi.to_gbk(line), end='')
 
