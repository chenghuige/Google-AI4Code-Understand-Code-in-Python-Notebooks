#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   /home/users/chenghuige/tools/json-print.py
#        \author   chenghuige  
#          \date   2013-11-24 21:05:31.482519
#   \Description  
# ==============================================================================

import sys,os
import json
#s = open(sys.argv[1]).read().decode('utf8')
#print(json.dumps(json.loads(s),sort_keys=True, indent=4, ensure_ascii=True).encode('utf8'))
s = open(sys.argv[1]).read()
print(json.dumps(json.loads(s),sort_keys=True, indent=4, ensure_ascii=True))
 
