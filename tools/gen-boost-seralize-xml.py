#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   /home/users/chenghuige/tools/gen-boost-seralize.py
#        \author   chenghuige  
#          \date   2014-09-07 16:42:04.643648
#   \Description  
# ==============================================================================

import sys,os

print '''
friend class boost::serialization::access;
template<class Archive>
void serialize(Archive &ar, const unsigned int version)
{'''
for line in sys.stdin:
    line = line[:line.rfind('=')]
    line = line[:line.find(';')]
    l = line.strip().split()
    if (len(l) > 1):
        if l[0].startswith('//'):
            continue
        print 'ar & BOOST_SERIALIZATION_NVP(' + l[-1] + ');' 
print '}'

 
