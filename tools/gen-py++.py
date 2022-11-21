#!/usr/bin/env python
import os
import sys, gflags
from pyplusplus import module_builder
FLAGS=gflags.FLAGS

gflags.DEFINE_string('i','','input files like: a.h,b.h,c.h')
gflags.DEFINE_string('m','','module_name like: libchg_py')
gflags.DEFINE_string('o','','out file like: chg_py.cc')

def run():
  file_li = FLAGS.i.split(',')
  print file_li
  mb = module_builder.module_builder_t(
    files = file_li,
    gccxml_path='/home/users/chenghuige/.jumbo/bin/gccxml')
  mb.build_code_creator( module_name=FLAGS.m )                        
  mb.code_creator.user_defined_directories.append( os.path.abspath('.') )
  mb.code_creator.user_defined_directories.append( os.path.abspath('./include') )
  mb.code_creator.user_defined_directories.append( os.path.abspath('./utils') )
  mb.write_module( os.path.join( os.path.abspath('.'), FLAGS.o ) )    

def main(argv):
  argv = FLAGS(argv)
  run()

if __name__ == '__main__':
  main(sys.argv)
