#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2015-02-26 17:48:25.559868
#   \Description  
# ==============================================================================

import sys,os
import time,datetime

pre = None
while(True):
  now = time.strftime('%Y%m%d',time.localtime(time.time())) 
  l = time.localtime()
  now_min = int(l[4])
  if now_min == 10 or now_min == 40:
    command = f'sh {sys.argv[1}'
    print('command:', command)
    os.system(command)
  time.sleep(20)



 
