#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   kill-port.py
#        \author   chenghuige  
#          \date   2019-09-26 00:48:31.686296
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import subprocess 

port = sys.argv[1]

#result = subprocess.check_output(["netstat", "-lnp", "|", "grep", port])
result = os.popen('netstat -lnp | grep {}'.format(port)).read()
pid = result.strip().split()[-1].split('/')[0]
command = 'kill {}'.format(pid)
print(command)
os.system(command)
