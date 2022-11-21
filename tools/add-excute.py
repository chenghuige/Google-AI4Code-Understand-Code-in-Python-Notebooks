#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   add-excute.py
#        \author   chenghuige  
#          \date   2011-05-05 22:29:55.526075
#   \Description  
# ==============================================================================

import sys,os,glob

def deal(file_name):
  file_name_ = file_name[:file_name.index('.')]
  out = open('CMakeLists.txt','a')
  res = """
ADD_EXECUTABLE(%s %s)               
TARGET_LINK_LIBRARIES(%s ${LIBS})                  
                                                   
    """%(file_name_, file_name, file_name_)        
  out.write(res)                                   
  out.close()                                      
  
if (sys.argv[1] == '.'):
  files = glob.glob("*.cpp")
  for file in files:
    deal(file)
else:
  deal(sys.argv[1]) 










