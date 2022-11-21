#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   select.py
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
df.to_excel(sys.argv[4], index = False)
