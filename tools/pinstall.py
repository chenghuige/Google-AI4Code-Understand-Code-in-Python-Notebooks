#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   pinstall.py
#        \author   chenghuige  
#          \date   2019-08-05 19:12:38.443688
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

input = ' '.join(sys.argv[1:])
#command = f'pip install {input} -i https://pypi.tuna.tsinghua.edu.cn/simple' 
command = f'pip install {input} -i https://mirrors.aliyun.com/pypi/simple' 
print(command)
os.system(command)
