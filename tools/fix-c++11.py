#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   c++11fix.py
#        \author   chenghuige  
#          \date   2014-04-16 16:19:59.330937
#   \Description  
# ==============================================================================

import sys,os
import re 

pre_nogccxml = 0
pre_rvalue = False
pre_construct = False
construct_count = 0
pre_define = False
pre_comment = 0
base_classes = set()
pre_static_ref_func = False
#---for enum clas
out = open('enum_class.txt', 'a')
pre_enum_class = False
class_name = ''

pre_template_line = ''
pre_template = 0

def print_(line, add_comment = True):
		global pre_template_line
		if pre_template_line == '':
				print line 
		else:
				if add_comment and line.startswith('//'):
						for item in pre_template_line.split('\n'):
							print '//' + item
				else: 
						print pre_template_line 
				print line
				pre_template_line = ''

input = sys.stdin
if len(sys.argv) > 1:
		input = open(sys.argv[1])
for line in input:
				line = line.strip()
	
				#------cereal shared ptr
				if line.startswith('CEREAL_REGISTER_TYPE'):
						print_('//' + line)
						continue
				
				#-------cereal to boost
				line = line.replace('cereal::', 'boost::serialization::')
				
				#------comment // /* */
				if line.startswith('//'):
						print_(line, False) 
						continue

				if line.startswith('/*'):
						if not line.endswith('*/'):
								pre_comment += 1 
						print_(line, False) 
						continue
				if pre_comment:
						if line.endswith('*/'):
								pre_comment -= 1
						print_(line, False) 
						continue

				#remove comment which is after code in the same line  x += 3;  //this is ...
				if line.find('//') > 0:
						line = line[:line.find('//')].rstrip()
				if line.endswith('*/'):
						if line.find('/*') > 0:
								line = line[:line.find('/*')].rstrip()
				
				#template<
				#        template .. >
				if not pre_template and line.startswith('template<') or line.startswith('template <'):
						pre_template_line = line 
						for item in line:
							if item == '<':
								pre_template += 1
							if item == '>':
								pre_template -= 1
						continue
				if pre_template:
						for item in line:
							if item == '<':
								pre_template += 1
							if item == '>':
								pre_template -= 1
						pre_template_line = pre_template_line + '\n' + line
						continue

				#---- hack boost, serialization
				if (line.find("boost/math/distributions/students_t.hpp") >= 0 or line.find("boost/date_time/posix_time/posix_time.hpp") >= 0):
						print_('//' + line)
						continue

				if (line.find('serialization/') >= 0 or line.find('serialization::') >= 0 or line.find('boost/archive/') >= 0):
						print_('//' + line)
						continue

				# hack define
				if line.replace(' ', '').startswith('#define') and line.endswith('\\'):
						pre_define = True 
						print_(line) 
						continue
				if pre_define:
						if not line.endswith('\\'):
								pre_define = False 
						print_(line) 
						continue

				#---c++11 override remove
				if line.endswith('override;'):
						line = line.replace('override', '')
						print_(line) 
						continue

				#---c++11 rvalue && remove
				if (line.find('&&') >= 0):
						if not line.endswith(';'):
								pre_rvalue = True
						print_('//' + line)
						continue
				#for && constructor
				if pre_rvalue:
						if line.startswith('}') or line.endswith(';'):
								pre_rvalue = False
						print_('//' + line)
						continue 
 
				#using Vector::Vector;
				l = [s for s in line.replace(',', ' ').replace(':', ' ').split(' ') if s != '']
				if len(l) >=  4 and (l[0] == 'class' or l[0] == 'struct') and l[2] == 'public':
						base_classes = set()
						for item in l[3:]:
								base_classes.add(item)

				if line.startswith('using') and len(base_classes) > 0:
						find = False 
						for item in base_classes:
								if line.endswith('%s::%s;'%(item, item)):
										find = True
										break
						if find:
								print_('//' + line)
								continue

				if line.startswith(':'):
						pre_construct = True
						construct_count = 0
						print_('//' + line)
						continue

				if pre_construct:
						if line.startswith('{'):
								construct_count += 1
						if line.startswith('}'):
								construct_count -= 1
								if construct_count == 0:
										pre_construct = False
										print_(';')
						continue

				if line.startswith('enum class'):
						l = line.split()
						class_name = l[2]
						out.write(class_name + '\n');
						if line.endswith('};'):
								line = line[line.find('{') + 1 : line.find('};')]
								member_names = [class_name + '__enum__' + (item + ' ')[:item.find('=')].strip() for item in line.split(',') if item != '']
								print_('enum ' + class_name) 
								print_('{')
								for member in member_names:
										print member + ','
								print_('};')
						else:
								pre_enum_class = True
								print_(line.replace('class', ''))
						continue
		
				if pre_enum_class:
						if line.endswith('};'):
								pre_enum_class = False 
						else:
								if line.endswith(','):
										print_((class_name + '__enum__' + line[:line.find('=')]).strip() + ',')
										continue
								elif not (line.startswith('{') or line == ''):
										print_((class_name + '__enum__' + (line + ' ')[:line.find('=')]).strip() + ',')
										continue

				#if line.startswith('enum class'):
				#    enum_class = line.split()[2]
				#    enum_classes.add(enum_class)
				#    head = 'class ' + enum_class + ' {  enum '
				#    if not line.endswith('};'):
				#        pre_enumclass = True
				#        print_(head)
				#    else:
				#        body = line[line.find('{'):]
				#        print_(head + body + '};')
				#    #print_('//' + line) 
				#    continue;

				#if pre_enumclass:
				#    if line.startswith('};'):
				#        pre_enumclass = False 
				#        print_('};')
				#    #print_('//' + line)
				#    print_(line) 
				#    continue
				
				#find = False
				#for item in enum_classes:
				#    if line.find(item) >= 0:
				#        find = True
				#        break 
				#if find:
				#    print_('//' + line)
				#    continue

				if (line.startswith('/') or line.startswith('#') or line.startswith('*') or line.startswith('namespace')):
						print_(line)
						continue 

				org_line = line
				p = line.find('//') # int a = 3; //lifds
				q = line.find('/*')
				if (q > p):
						p = q
				if (p >= 0):
						line = line[:p]
				end_idx = line.rfind(';')
				if (end_idx == -1):
						print_(org_line)
						continue
				if (line.startswith('const static') or line.startswith('static')):
						print_(org_line)
						continue
	
	
				#line = line[:end_idx] #comment里面不要同行出现)
                                #@TODO will fail for below, so move to h2cc will be better
                                # void abc(int x,
                                #    int && y)
				if line.find('&&') >= 0:
						print_('//' + line)
						continue

				#@FIXME will cause problem..
				#if (line.find('ostream&') >= 0 or line.find('fstream&') >= 0) and line.find(cout) >= 0 and line.endswith(';'):
				#        print_('//' + line) 
				#        continue
	
				if (line.find('= default') >= 0):
						print_((line)[:line.rfind('=')].strip() + ';')
						continue

				idx1 = line.rfind(')')
				idx2 = line.rfind('=')
				idx3 = line.find('(')
				idx4 = line.find('=')
				
				line2 = line.replace(' ', '')
				if idx3 >= 0 and idx4 >= 0 and idx3 > idx4 and line2.find('operator=') < 0 and line.find('operator>=') < 0 and line.find('operator<=') < 0 and line.find('operator+=') < 0:
						line = line[:idx4] + ';'
						if line.startswith('const'):
								line = line[len('const') + 1:]
						print_(line) 
						continue

				if (idx1 > idx2):
						print_(org_line)
						continue

				if idx2 == -1:
						print_(org_line)
						continue 
				line = line[:idx2]
				line = line.strip() + ';'
				if line.startswith('const'):
						line = line[len('const') + 1:]
						print_(line)


 
