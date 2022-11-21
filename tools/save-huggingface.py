#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../../../../utils')
import gezi

# In[3]:
backbone_name = sys.argv[1]

gezi.save_huggingface(backbone_name, '/work/data/huggingface')
