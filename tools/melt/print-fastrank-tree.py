#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   print-fastreank-tree.py
#        \author   chenghuige  
#          \date   2014-10-04 00:34:59.825146
#   \Description  
# ==============================================================================

import sys,os
from gflags import *
from gezi import *
from libmelt import *
from BinaryTree import *
from TreeWriter import *

DEFINE_string('model', './model/model.json', '')
DEFINE_string('feature', '', '')
DEFINE_integer('tree', -1, '-1 means print all trees')
DEFINE_boolean('use_invisable_node', False, '')
DEFINE_string('outfile', 'tree.png', '')
DEFINE_string('dir', './trees', 'if --tree = 0 save to this dir all the tee pic')

_final_score = '0'
def get_tree_(node_idx, fe, tree, fnames, is_inpath):
	node = Node()
	node.attr = {'color' : ".7 .3 1.0", 'style' : 'filled'}
	node.leftEdgeAttr = {'color' : 'blue', 'penwidth' : '2.5', 'label' : '<='}
	node.rightEdgeAttr = {'color' : 'green', 'penwidth' : '2.5', 'label' : '>'}
	if is_inpath:
		node.attr['color'] = '#40e0d0'
	if node_idx < 0:
		node.attr['shape'] = 'box'
		node.attr['label'] = str(tree.leafValue[-node_idx - 1])
		if is_inpath:
			global _final_score
			print FLAGS.tree, node.attr['label']
			_final_score = node.attr['label']
		return node
	name = fnames[tree.splitFeature[node_idx]]
	label = '%s\l%f <= %f?\l[%f]'%(name, fe[tree.splitFeature[node_idx]], tree.threshold[node_idx], tree.previousLeafValue[node_idx])
	node.attr['label'] = label 
	if is_inpath:
		l = fe[tree.splitFeature[node_idx]] <= tree.threshold[node_idx]
		r = 1 - l
		if l:
			node.leftEdgeAttr['color'] = 'black'
		else:
			node.rightEdgeAttr['color'] = 'black'
	else:
		l = r = 0
	node.left = get_tree_(tree.lteChild[node_idx], fe, tree, fnames, l)
	node.right = get_tree_(tree.gtChild[node_idx], fe, tree, fnames, r)
	return node 

def get_tree(model, fe, index):
	tree = model.trees[index]
	fnames =  model.Predictor.featureNames
	btree	= BinaryTree()
	node_idx = 0
	btree.root = get_tree_(node_idx, fe, tree, fnames, 1)
	return btree	

def write_tree(writer, tree, outfile):
	writer.tree = tree
	writer.Write(outfile)

def main(argv):
	try:
		argv = FLAGS(argv)  # parse flags
	except gflags.FlagsError, e:
		print '%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS)
		sys.exit(1)

	model = jsonfile2obj(FLAGS.model) 
	fe = Vector(FLAGS.feature)

	writer = TreeWriter()
	if FLAGS.use_invisable_node:
		writer.use_invisable_node = True

	if FLAGS.tree >= 0:
		tree = get_tree(model, fe, FLAGS.tree)
		write_tree(writer, tree, FLAGS.outfile)
	else:
		os.system('mkdir -p %s'%FLAGS.dir)
		num_trees = len(model.trees)
		score = 0
		for i in range(num_trees):
			FLAGS.tree = i
			tree = get_tree(model, fe, i)
			outfile = "%s/%s-%d.png"%(FLAGS.dir, _final_score, i)
			write_tree(writer, tree, outfile)
			score += float(_final_score)
		print 'output: ' + str(score)
	
if __name__ == "__main__":  
	 main(sys.argv)  

 
