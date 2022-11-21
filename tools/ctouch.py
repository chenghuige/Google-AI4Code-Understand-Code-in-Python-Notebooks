#!/usr/bin/env python
# -*- coding: gbk -*-
#this is for helping creating c++ standard .cc files
#especially for small testing programs or acm programs 
#for single .cc file

import sys,os,datetime

def usage():
    """ ctouch.py  compress.cc   
    """

def run(argv):
    file_name = argv[0]
    
    if file_name in os.listdir('./'):
        print("The file you want to create already exsits")
        return
    
    #create file
    command = "touch " + file_name
    os.system(command)
    
    #open file
    f = open(file_name, 'w')
    
    #write file
    #write file info
    date_time = datetime.datetime.now()
    
    file_info = r"""/** 
 *  ==============================================================================
 * 
 *          \file   %s
 *
 *        \author   chenghuige  
 *
 *          \date   %s
 *  
 *   \Description  
 *
 *  ==============================================================================
 */
"""%(file_name, date_time) 

    f.write(file_info)
    f.write('\n')
        
    #----------------------------------------------------    
    content = """#include <iostream>
#include <stdio.h>
using namespace std;

int main(int argc, char *argv[])
{
    
  return 0;
}
"""
    f.write(content)
    f.close()
    file_name_ = file_name[:file_name.index('.')]
    f3 = open('COMAKE', 'a')
    f3.write("Application(\'%s\',Sources(\'%s\',srcs), OutputPath(\'./bin\'))\n" % (file_name_, file_name))
    f3.close()
    command = "svn add %s"%(file_name)
    os.system(command)
    command = "win-path.py %s"%(file_name)
    os.system(command)

#----------------------------------------------------------
if __name__ == '__main__':
    run(sys.argv[1:]) 
