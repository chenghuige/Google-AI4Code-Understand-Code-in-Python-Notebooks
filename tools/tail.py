#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tail.py
#        \author   chenghuige  
#          \date   2019-10-25 21:09:28.802012
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import time

while True:
  os.system('tail -f %s' % sys.argv[1])
  
  
