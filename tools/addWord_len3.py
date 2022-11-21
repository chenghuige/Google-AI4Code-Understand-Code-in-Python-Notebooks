#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   findNewWord_len3.py
#        \author   chenghuige  
#          \date   2011-01-12 14:08:39.446283
#   \Description  findNewWord_len3.py   common.txt  hot_phrase_len3.txt //
# ==============================================================================

import sys
 
def run(input1, input2):
    dict = {}
    for line in open(input1):
        line = line.strip()
        dict[line] = 0

    for line in open(input2):
        line = line.strip()
        if ( (line not in dict) and (not (line[:2] in dict or line[4:6] in dict or line[:4] in dict or line[2:6] in dict)) ):
            dict[line] = 0

    out = open(input1, 'w')

    l = sorted(dict.items(), key=lambda dict:dict[0])

    for k,v in l:
        out.write(k + '\n')

if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    run(input1, input2)
  
