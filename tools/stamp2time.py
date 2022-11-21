#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   stamp2time.py
#        \author   chenghuige  
#          \date   2019-09-11 11:11:21.469291
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import time

print(time.localtime(int(sys.argv[1])))

