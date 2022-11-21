#!/usr/bin/env python
import sys,os,datetime,subprocess

def run(argv):
  file_name = argv[0]                                                            

  if file_name in os.listdir('./'):
      print("The file you want to create already exsits")
      return                                                                     
                                                                                 
  #create file                                                                   
  #command = "touch-file " + file_name
  #os.system(command)
  #subprocess.Popen(["powershell", command])

  #open file
  f = open(file_name, 'w')

  #write file
  #write file info
  date_time = datetime.datetime.now()

  file_info = r"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   %s
#        \author   chenghuige  
#          \date   %s
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from absl import flags
FLAGS = flags.FLAGS

import sys 
import os

  """%(file_name, date_time) 


  f.write(file_info)
  f.write('\n')

  #----------------------------------------------------    
  #content = r"""if __name__ == '__main__':
  #"""
  #f.write(content)
  #f.write('\n')
  f.close()
  #os.system('svn add %s'%file_name)
  #os.system('git add %s'%file_name)
  #os.system('win-path.py %s'%file_name)
  os.system('chmod 777 %s'%file_name)

#----------------------------------------------------------
if __name__ == '__main__':
    run(sys.argv[1:]) 
