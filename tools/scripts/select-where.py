#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   where.py
#        \author   chenghuige  
#          \date   2014-01-03 21:56:08.497282
#   \Description  
# ==============================================================================

import sys
from pandasql import *
import pandas as pd
 
pysqldf = lambda q: sqldf(q, globals())
 
q  = """
SELECT
    %s
FROM
    table_
WHERE
    %s;
"""%(sys.argv[2], sys.argv[3])

table_ = pd.read_table(sys.argv[1], encoding='utf-8')
df = pysqldf(q)    

names = ['Instance','Label','True','Assigned']
for name in names:
  try:
    df[name] = df[name].astype(int)
  except Exception:
    pass
  name = name.lower()
  try:
    df[name] = df[name].astype(int)
  except Exception:
    pass
try:
	df[u'标注'] = df[u'标注'].astype(int)
except Exception:
	pass

if len(sys.argv) > 4:
	df.to_csv(sys.argv[4], index = False, sep ='\t')

from StringIO import StringIO
import prettytable    

output = StringIO()
df.to_csv(output, index = False, sep ='\t')
output.seek(0)

pt = prettytable.from_csv(output)
print pt 

