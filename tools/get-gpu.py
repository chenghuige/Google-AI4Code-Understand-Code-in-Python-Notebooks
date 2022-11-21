#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   choose-gpu.py
#        \author   chenghuige  
#          \date   2020-02-16 15:39:41.908457
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import os 

import gezi

gpus = gezi.get_gpus()

print(gpus[0])

