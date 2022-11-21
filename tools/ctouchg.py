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
        
    #----------------------------------------------------    
    content = """#define private public
#define protected public
#include "common_util.h"

using namespace std;
using namespace gezi;

DEFINE_int32(vl, 0, "vlog level");
DEFINE_int32(level, 0, "min log level");
DEFINE_string(type, "simple", "");
DEFINE_bool(perf,false, "");
DEFINE_int32(num, 1, "");
DEFINE_string(i, "", "input file");
DEFINE_string(o, "", "output file");

void run()
{

}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::SetVersionString(get_version());
    int s = google::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_log_dir.empty())
        FLAGS_logtostderr = true;
    FLAGS_minloglevel = FLAGS_level;
    //LogHelper::set_level(FLAGS_level);
    if (FLAGS_v == 0)
        FLAGS_v = FLAGS_vl;

    run();

   return 0;
}
"""
    f.write(content)
    f.close()
    
    command = "svn add %s"%(file_name)
    os.system(command)
    file_name_ = file_name[:file_name.index('.')]
    res = """
ADD_EXECUTABLE(%s %s)
TARGET_LINK_LIBRARIES(%s ${LIBS})
    
    """%(file_name_, file_name, file_name_)
    f3 = open('COMAKE', 'a')
    f3.write("#Application(\'%s\',Sources(\'%s\',srcs, ENV.CppFlags()+CppFlags('-O3 -DNDEBUG')))\n" % (file_name_, file_name))
    f3.write("Application(\'%s\',Sources(\'%s\',srcs))\n" % (file_name_, file_name))
    f3.close()
    command = "win-path.py %s"%(file_name)
    os.system(command)
#----------------------------------------------------------
if __name__ == '__main__':
    run(sys.argv[1:]) 

