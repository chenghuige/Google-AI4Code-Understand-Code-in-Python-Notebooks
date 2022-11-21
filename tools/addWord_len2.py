#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   findNewWord_len2.py
#        \author   chenghuige  
#          \date   2011-01-12 14:08:32.448702
#   \Description   findNewWord_len2.py   common.txt  hot_phrase_len2.tx  
# ==============================================================================

import sys
 
def run(input1, input2):
    dict = {}
    for line in open(input1):
        line = line.strip()
        dict[line] = 0

    for line in open(input2):
        line = line.strip()
        if ( (line not in dict) and (not (line[:2] in dict and line[2:4] in dict)) ):
            dict[line] = 0

    out = open(input1, 'w')

    l = sorted(dict.items(), key=lambda dict:dict[0])

    for k,v in l:
        out.write(k + '\n')

if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    run(input1, input2)
