#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   remove-files.py
#        \author   chenghuige  
#          \date   2014-11-26 15:37:43.592367
#   \Description  
# ==============================================================================

 
import sys,os
import stat
def sortdir(path, sort_cond = 'mtime', sort_filter = None, reverse = True, abspath = True, onlyfn = True):
	'''
	sort dir by give condition & filter
	:param path: dir path
	:param sort_cond:
	ctime: create time
	mtime: last modify time
	atime: atime
	size: file size
	:param sort_filter:
	function to filter
	1: only file
	2: only dir
	3: both file and dir
	func: custom function
	:param reverse:
	if sort reversed
	:param abspath:
	if True, return list with absolute path of file
	or else, return relative
	:param onlyfn:
	if True, return [filename1, filename2, ....] at sort_cond
	else, return [(filename, os.stat(file), (), ...] at sort_cond
	:return:
	[(filename, os.stat(file), (), ...] at sort_cond
	'''
	if sort_cond == "mtime":
		f_sort_cond = lambda e:e[1].st_mtime
	elif sort_cond == "ctime":
		f_sort_cond = lambda e:e[1].st_ctime
	elif sort_cond == "atime":
		f_sort_cond = lambda e:e[1].st_atime
	elif sort_cond == "size":
		f_sort_cond = lambda e:e[1].st_size
	else:
		f_sort_cond = lambda e:e[1].st_mtime
		f_sf = None
	if sort_filter == None or sort_filter == 3:
		f_sf = None
	elif type(sort_filter) == type(lambda x:x):
		f_sf = sort_filter
	else:
		if sort_filter == 1:
			f_sf = lambda e: stat.S_ISDIR(e.st_mode) == 0
		elif sort_filter == 2:
			f_sf = lambda e: stat.S_ISDIR(e.st_mode)
		else:
			f_sf = None
	if onlyfn:
		return map(lambda e:e[0], __sortdir(path, f_sort_cond, f_sf, reverse, abspath))
	return __sortdir(path, f_sort_cond, f_sf,reverse, abspath)

def __sortdir(path, sort_cond, sort_filter, reverse, abspath):
	fns = os.listdir(path)
	if not path.endswith('/'):
		path = path + '/'
	a_fns = map(lambda f: path+f, fns)
	sts = map(os.stat, a_fns)
	if abspath:
		res = zip(a_fns, sts)
	else:
		res = zip(fns, sts)
	if sort_filter == None:
		return sorted(res, key = sort_cond, reverse = reverse)
	n_res = []
	for e in res:
		if sort_filter(e[1]):
			n_res.append(e)
	return sorted(n_res, key = sort_cond, reverse = reverse)

def main(argv):
	print argv[1]
	print sortdir(argv[1])

if __name__ == "__main__":  
	 main(sys.argv)  