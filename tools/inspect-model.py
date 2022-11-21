#!/usr/bin/env python 
import sys
import h5py
import numpy as np
from icecream import ic

# TODO 递归打印
def print_keras_weights(weight_file_path):
  f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
  ic(weight_file_path)
  if len(f.attrs.items()):
    ic(f.attrs['layer_names'])
    for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
      ic(layer)
    #   for key, value in g.attrs.items():
    #     ic(key, value)

      for name, d in g.items(): # Read the Dataset class that stores specific information in each layer
        for k, v in d.items():
          try:
            ic(k, v.shape)
          except Exception:
            # ic(k)
            try:
              for k, v in v.items():
                try:
                  ic(k, v.shape)
                except Exception:
                  ic(k)
                  for k, v in v.items():
                    try:
                      ic(k, v.shape)
                    except Exception:
                      ic(k)
                      for k, v in v.items():
                        try:
                          ic(k, v.shape)
                        except Exception:
                          ic(k)
                          for k, v in v.items():
                            try:
                              ic(k, v.shape)
                            except Exception:
                              ic(k)
                              for k, v in v.items():
                                try:
                                  ic(k, v.shape)
                                except Exception:
                                  ic(k)
                                  for k, v in v.items():
                                    try:
                                      ic(k, v.shape)
                                    except Exception:
                                      ic(k)
                               
            except Exception:
              pass

print_keras_weights(sys.argv[1])

