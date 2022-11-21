#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   cmake2make.py
#        \author   chenghuige  
#          \date   2011-06-02 16:03:53.128457
#   \Description  
# ==============================================================================

input = 'CMakeLists.txt' 
output = 'Makefile'
out = open(output, 'w')
head = """OUTPUT=$(patsubst %.cc,%, $(wildcard *.cc))
CC = g++
CPPFLAGS = -g -O3 -finline-functions -pipe $(INCLUDES)
"""
out.write(head)

find_set = 0
find_include = 0
find_link = 0
find_libs = 0

def convert_set(str):
  u = str[str.find('(') + 1: str.find(')')]
  li =[]
  for i in range(len(u)):
    if (u[i].isspace()):
      li.append('=')
    elif u[i] == '{':
      li.append('(')
    elif u[i] == '}':
      li.append(')')
    else:
      li.append(u[i])
  return ''.join(li)
def convert_inlcude_link(str, pre):
  li = []
  for i in range(len(str)):
    if (str[i]=='$'):
      li.append(pre+'$')
    elif (str[i] == '{'):
      li.append('(')
    elif (str[i] == '}'):
      li.append(')')
    else:
      li.append(str[i])
  li.append('\\')
  return ''.join(li)

def convert_libs(str):
  li = str.split()
  for i in range(len(li)):
    li[i] = '-l' + li[i]
  return ' '.join(li) + '\\' 

with open(input) as f:
  m = f.readlines()
  for j in range(10):
    if (m[j].lower().startswith('project(')):
      project = m[j][m[j].find('(') + 1: m[j].find(')')]
      out.write("%s_SOURCE_DIR=./\n"%project)
      
  i = 20
  while i < len(m):
    if (find_libs):
      break
    line = m[i].strip()
    if (not find_set and not line.lower().startswith('set')):
      i += 1
      continue
    elif not find_set:
      while (line.lower().startswith('set')):
        result = convert_set(line)
        #print 'set: ' + result
        out.write(result + '\n')
        i += 1
        line = m[i].strip()
      i += 1
      find_set = 1
    if (not find_include and not line.lower().startswith('include_directories')):
      i += 1
      continue
    elif not find_include:
      i += 1
      line = m[i].strip()
      out.write("\nINCLUDES = ")
      while(line.startswith('$')):
        result = convert_inlcude_link(line, '-I')
        #print 'include: ' + result
        out.write(result + '\n')
        i += 1
        line = m[i].strip()
      i += 1
      find_include = 1
    if (not find_link and not line.lower().startswith('link_directories')):
      i += 1
      continue
    elif not find_link:
      i += 1
      line = m[i].strip()
      out.write("\nLDFLAGS = ")
      while(line.startswith('$')):
        result = convert_inlcude_link(line, '-L')
        #print 'include: ' + result
        out.write(result + '\n')
        i += 1
        line = m[i].strip()
      i += 1
      find_link = 1
    if (not find_libs and not line.lower().startswith('set(libs')):
      i += 1
      continue
    elif not find_libs:
      i += 1
      line = m[i].strip()
      while(not line.startswith(')')):
        result = convert_libs(line)
        out.write(result + '\n')
        #print 'libs: ' + result
        i += 1
        line = m[i].strip()
      i += 1
      find_libs = 1


tail = """
all	: $(OUTPUT) 
	if [ ! -d output ]; then mkdir output; fi
	cp $(OUTPUT) output
	rm -f *.o

$(OUTPUT): %: %.o
	$(CC)  -o $@ $< $(INCLUDES) $(LDFLAGS)  
%.o	: %.cc
	$(CC) $(CPPFLAGS) -c $< -o $@ $(INCLUDES) 

clean:
	rm -f *.o $(OUTPUT)
	rm -rf output
"""

out.write(tail)
