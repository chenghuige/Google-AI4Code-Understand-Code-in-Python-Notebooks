#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   hist-feature.py
#        \author   chenghuige  
#          \date   2014-10-02 16:22:18.533077
#   \Description  
# ==============================================================================

from gezi import * 

DEFINE_boolean('norm', False, "")
DEFINE_integer('bins', 10, '')

def main(argv):
	try:
		argv = FLAGS(argv)  # parse FLAGS
	except gflags.FlagsError, e:
		print '%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS)
		sys.exit(1)
	input = argv[1]
	
	input1 = input.replace('.txt', '.pos.txt')
	input2 = input.replace('.txt', '.neg.txt')
	print input1, input2
	bins_ =  FLAGS.bins
	print bins_
	if not os.path.exists(input1) or not os.path.exists(input2):
		cmd = 'melt %s -c sd -header 1 -label 1'%(input)
		print cmd
		os.system(cmd)
	
	input12 = input1.replace('.txt', '.csv')
	
	if not os.path.exists(input12):
		cmd = 'melt2csv.py %s > %s'%(input1, input12)
		print cmd
		os.system(cmd)
	
	input22 = input2.replace('.txt', '.csv')
	if not os.path.exists(input22):
		cmd = 'melt2csv.py %s > %s'%(input2, input22)
		print cmd
		os.system(cmd)
	
	input1 = input12
	input2 = input22
	
	print input1,input2 
	print 'loading: ' + input1
	tp = pd.read_table(input1)
	print 'loading: ' + input2
	tn = pd.read_table(input2)
	
	cmd = 'mkdir -p ./hist'
	print cmd
	os.system(cmd)
	
	prefix = input[:input.rindex('.txt')]
	
	col = argv[2]
	file_name = './hist/' + prefix + '.' + col + '.png'
	print file_name
	plt.hist((tp[col], tn[col]), bins=bins_, normed=FLAGS.norm, color=('crimson', 'chartreuse'), label=('1','0'))
	plt.legend()
	plt.savefig(file_name)
		
	
if __name__ == "__main__":  
	 main(sys.argv)  
