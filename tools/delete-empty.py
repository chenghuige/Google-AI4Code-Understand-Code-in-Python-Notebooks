#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   delete-empty.py
#        \author   chenghuige  
#          \date   2020-02-27 09:08:23.383813
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

root = './'
if len(sys.argv) > 1:
  root = sys.argv[1]

files = os.listdir(root)  # 获取路径下的子文件(夹)列表
for file in files:
    if os.path.isdir(file):  # 如果是文件夹
        if not os.listdir(file):  # 如果子文件为空
           print('empty dir:', file)
           os.rmdir(file)  # 删除这个空文件夹
    elif os.path.isfile(file):  # 如果是文件
        if os.path.getsize(file) == 0:  # 文件大小为0
           print('empty file:', file)
           os.remove(file)  # 删除这个文件
  
