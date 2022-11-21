#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   delete-old-models.py
#        \author   chenghuige  
#          \date   2020-02-27 17:31:20.479046
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob
import melt

root = sys.argv[1] if len(sys.argv) > 1 else './'
latest_checkpoint = melt.latest_checkpoint(root)
if latest_checkpoint:
  latest_checkpoint_name = os.path.basename(latest_checkpoint)
  for file in glob.glob(f'{root}/model.ckpt*'):
    if not os.path.basename(file).startswith(latest_checkpoint_name):
      print('deleting', file) 
      os.system(f'rm -rf {file}')

