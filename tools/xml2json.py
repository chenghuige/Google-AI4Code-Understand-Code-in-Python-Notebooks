#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   xml2json.py
#        \author   chenghuige  
#          \date   2014-09-27 21:18:07.564466
#   \Description  
# ==============================================================================

import sys,os

import xmltodict, json

doc = xmltodict.parse(open(sys.argv[1]), process_namespaces=True)
print json.dumps(doc)
