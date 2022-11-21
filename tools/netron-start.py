#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   netron.py
#        \author   chenghuige  
#          \date   2021-01-01 22:35:20.652442
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import netron
import socket
host = socket.gethostname()
try:
  host = socket.gethostbyname(host)
except Exception:
  host = 'localhost'
port = 8822
if len(sys.argv) > 2:
  port = int(sys.argv[2])
netron.start(sys.argv[1], port=port, host=host)

