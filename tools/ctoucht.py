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
 *  \Description:
 *
 *  ==============================================================================
 */
"""%(file_name, date_time) 

    f.write(file_info)
    f.write('\n')
    #print file_name
    #file_name_ = file_name[file_name.index('/') + 1:file_name.index('.')]
    file_name_ = file_name[:file_name.index('.')]
    #----------------------------------------------------    
    content = """#define _DEBUG
#define private public
#define protected public
#include "common_util.h"

using namespace std;
using namespace gezi;
DEFINE_int32(vl, 5, "vlog level");
DEFINE_string(i, "", "input");
DEFINE_string(o, "", "output");
DEFINE_string(type, "simple", "");

TEST(%s, func)
{

}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  int s = google::ParseCommandLineFlags(&argc, &argv, false);
  if (FLAGS_log_dir.empty())
    FLAGS_logtostderr = true;
  if (FLAGS_v == 0)
    FLAGS_v = FLAGS_vl;
  
  return RUN_ALL_TESTS();
}
"""%(file_name_.replace('test_',''))
    f.write(content)
    f.close()
    f2 = open('COMAKE', 'a')
    f2.write("Application(\'%s\',Sources(\'%s\', srcs))\n" % (file_name_, file_name))
    #f2.write("Application(\'%s\',Sources(\'%s\'))\n" % (file_name_, file_name))
    f2.close()
    os.system('svn add %s'%file_name)
    command = "win-path.py %s"%(file_name)
    os.system(command)
#----------------------------------------------------------
if __name__ == '__main__':
    run(sys.argv[1:]) 
