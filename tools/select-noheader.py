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
    table_;
"""%(sys.argv[2])

table_ = pd.read_table(sys.argv[1], encoding='utf-8')
df = pysqldf(q)    
print df.describe()
df.to_csv(sys.argv[3], index = False, sep='\t', header = False)

 
