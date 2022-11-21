#!/usr/bin/env python
#coding=gbk
import sys,os,datetime

def run(argv):
  file_name = argv[0]
  rfile = argv[1]
  
  if file_name in os.listdir('./'):
      print("The file you want to create already exsits")
      return                                                                     
                                                                                 
  #create file                                                                   
  command = "touch " + file_name
  os.system(command)
  
  #open file
  f = open(file_name, 'w')
  
  #write file
  #write file info
  date_time = datetime.datetime.now()
  
  file_info = r"""#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   %s
#        \author   chenghuige  
#          \date   %s
#   \Description  
# ==============================================================================

from pyper import *
 """%(file_name, date_time) 
 
  f.write(file_info)
  f.write('\n')
    
  #----------------------------------------------------    
  content = r"""if __name__ == '__main__':
    runR("source('%s')")
  """%(rfile)
  f.write(content)
  f.close()

#----------------------------------------------------------
if __name__ == '__main__':
    run(sys.argv[1:]) 
