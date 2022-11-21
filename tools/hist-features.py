#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   hist-features.py
#        \author   chenghuige  
#          \date   2014-10-02 16:22:18.533077
#   \Description   
# ==============================================================================

from gezi import * 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import glob 
import pandas as pd

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('norm', False, "")
flags.DEFINE_integer('bins', 10, '')
flags.DEFINE_string('col', '', '')
flags.DEFINE_boolean('header', False, "")
flags.DEFINE_string('command', '', 'command for melt')

def add_filename_suffix(s, suffix):
  if s.endswith('.txt'):
    return s.replace('.txt', ".%s.txt"%suffix)
  else:
    return s + '.' + suffix

def replace_filename_suffix(s, suffix):
  if s.endswith('.txt'):
    return s.replace('.txt', ".%s"%suffix)
  else:
    return s + '.' + suffix

def hist_col(col, label_tables, labels):
  plt.clf()
  file_name = './hist/' + col.replace(':', '_') + '.png'
  print file_name
  try:
    cols = [table[col] for table in label_tables]
    plt.hist(cols, bins=FLAGS.bins, normed=FLAGS.norm, label=labels)
    plt.legend()
    plt.savefig(file_name)	
  except Exception:
    print 'failed histogram print for col ', col

def main(argv):
  try:
    argv = FLAGS(argv)  # parse FLAGS
  except gflags.FlagsError, e:
    print '%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS)
    sys.exit(1)
  
  input = argv[-1]

  cmd = 'rm -rf ./hist-input; mkdir ./hist-input'
  print cmd
  os.system(cmd)

  cmd = 'cp %s ./hist-input'%input
  print cmd
  os.system(cmd)

  input = './hist-input/%s'%input

  first_line = open(input).readline()
  if not first_line.startswith('#'):
    print 'no header, generate fake header first'
    cmd = 'mlt %s %s -c addheader'%(input, FLAGS.command) 
    print cmd
    os.system(cmd)
    input = add_filename_suffix(input, 'addheader')

  cmd = 'mlt %s %s -c sd'%(input, FLAGS.command)
  print cmd
  os.system(cmd)

  label_files = []
  for ifile in glob.glob('./hist-input/*label*'):
    if not ifile.endswith('.csv'):
      ofile = replace_filename_suffix(ifile, 'csv')
      label_files.append(ofile)
      cmd = 'melt2csv.py %s > %s'%(ifile, ofile)
      print cmd
      os.system(cmd)
  

  cmd = 'mkdir -p ./hist'
  print cmd
  os.system(cmd)

  label_tables = []
  for label_file in label_files:
    table = pd.read_table(label_file)
    label_tables.append(table)
  
  labels = [str(i) for i in xrange(len(label_tables))]
  
  if FLAGS.col == '':
    for col in label_tables[0].columns:
      hist_col(col, label_tables, labels)
  else:
    col = FLAGS.col
    hist_col(col, label_tables, labels)

    
if __name__ == "__main__":  
   main(sys.argv)  

  
