#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   tlcmodel2melt.py
#        \author   chenghuige  
#          \date   2016-07-05 21:37:35.977873
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys,os
import libmelt 

pf = libmelt.PredictorFactory()
p = pf.LoadTextPredictor(sys.argv[1])

p.Save(sys.argv[1])
  
