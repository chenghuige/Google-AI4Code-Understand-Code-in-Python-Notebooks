#!/usr/bin/env python
# -*- coding: gbk -*-

import sys

import glob
import os

def run():
    file_name = 'COMAKE'
    
    #open file
    f = open(file_name, 'w')
    
    #----------------------------------------------------    
    head = """#edit-mode: -*- python -*-
#coding:gbk

WORKROOT('/home/work/chenghuige')

CopyUsingHardLink(True)

CPPFLAGS('-D_GNU_SOURCE -D__STDC_LIMIT_MACROS -DVERSION=\\\"1.9.8.7\\\"')

CFLAGS('-g -O3 -pipe -W -Wall -fPIC')

CXXFLAGS('-g -O3 -pipe -W -Wall -fPIC')

IDLFLAGS('--compack')

UBRPCFLAGS('--compack')

INCPATHS('./include ./utils /home/work/chenghuige/project/utils')

#LIBS('./libabc.a')


LDFLAGS('-lpthread -lcrypto -lrt')

srcs=GLOB('./src/*.cpp')

"""
    f.write(head)
    #try:
    if ('libs.txt' in os.listdir('./')):
        configs_f = open('libs.txt')
        configs_content = configs_f.readlines()
        libs = []
        for i in range(len(configs_content)):
            line = configs_content[i].strip()
            if line.endswith(':'):
                module = line[:-1]
                i += 1
                line = configs_content[i].strip()
                names = line.split()
                for name in names:
                    libs.append('%s/%s' % (module, name))
                i += 1
            else:
                i += 1
        f.write("CONFIGS(\'%s\')\n"%' '.join(libs))
   
    f.write('\n')
    for file_name in glob.glob('*.cc'):
        f.write("Application(\'%s\',Sources(\'%s\', srcs), OutputPath(\'./bin\'))\n" % (file_name[:-3], file_name))
    
    tail = """
#StaticLibrary('sim-computing',Sources(user_sources),HeaderFiles(user_headers))
#SharedLibrary('sim-computing',Sources(user_sources),HeaderFiles(user_headers))
#Directory('demo')    
"""
    f.write(tail)
    f.close()

#----------------------------------------------------------
if __name__ == '__main__':
    run() 
