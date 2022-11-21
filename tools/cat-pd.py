#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   head-pd.py
#        \author   chenghuige  
#          \date   2019-12-23 07:01:45.406672
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import gezi
import pandas as pd 
from tabulate import tabulate

sep = gezi.guess_sep(open(sys.argv[1]).readline().strip())
df = pd.read_csv(sys.argv[1], sep=sep)
if len(sys.argv) > 2:
  cols = sys.argv[2].split(',')
  df = df[cols]

print(tabulate(df, headers=list(df.columns), tablefmt='psql'))
