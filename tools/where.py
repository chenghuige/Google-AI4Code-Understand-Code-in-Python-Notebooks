#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   where.py
#        \author   chenghuige  
#          \date   2014-01-03 21:56:08.497282
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import six 
if six.PY2:
  from io import BytesIO as IO
else:
  from io import StringIO as IO 

from pandasql import *
import pandas as pd
 
pysqldf = lambda q: sqldf(q, globals())
 
q  = """
SELECT
		*
FROM
		table_
WHERE
		%s;
"""%(sys.argv[2])

print(q)

table_ = pd.read_csv(sys.argv[1])

df = pysqldf(q)

#print('table', table_)
#print('df', df)
import prettytable    

output = IO()
df.to_csv(output, index=False)
output.seek(0)

if len(sys.argv) <= 3:
  # TODO could not determin delimiter
  #pt = prettytable.from_csv(output)
  #print(pt)
	#print(df)
	pass
else:
  # TODO.. not same as read.. for string wtih quote
	df.to_csv(sys.argv[3], index=False)

