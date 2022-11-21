#!/usr/bin/env python
# ==============================================================================
#          \file   json-print.py
#        \author   chenghuige  
#          \date   2013-11-24 21:05:31.482519
#   \Description  
# ==============================================================================

import sys,os
import json

print(json.dumps(json.load(sys.stdin),sort_keys=True, indent=4, ensure_ascii=True))
 
