#!/usr/bin/env python
# -*- coding: utf-8 -*-
#this is for helping creating c++ .h files
#adding ifndef define endif
#and namespace if there is also adding
#some file info lik file name , author, date

import sys, os, datetime

def usage():
    """ touch.py compressor.h  #this will add file info and ifndef define endif
        touch.py compressor.h compress  inter_compress #this will also add namespace compress and inter_compress
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
 *  ==============================================================================
 */
""" % (file_name, date_time)
    f.write(file_info)
    f.write('\n')
    l = []
    for i in range(len(file_name)):
        if (i > 0 and file_name[i].isupper()):
            l.append('_')
        l.append(file_name[i])
    define_name = ''.join(l)
    define_name = define_name.replace('/','_').replace('.','_').upper() + '_' #like compressor.h -> COMPRESSOR_H_
    f.write("#ifndef " + define_name + '\n')
    f.write("#define " + define_name + '\n')
    f.write('\n')
    for i in range(1, len(argv)):
        f.write("namespace " + argv[i] + ' {' + '\n')               #like namespace compress {
        #f.write('\n')
    f.write('\n')
    class_name = file_name[file_name.rfind('/') + 1: file_name.rfind('.')]
    if (class_name[0].isupper()):
      class_content = """class $name$ 
{
public:
	~$name$() = default;
	$name$() = default;
	$name$($name$&&) = default;
	$name$& operator = ($name$&&) = default;
	$name$(const $name$&) = default;
	$name$& operator = (const $name$&) = default;
public:

protected:
private:

};

""".replace('$name$',class_name)
      f.write(class_content);
    for i in range(1, len(argv)):
        f.write("}  //----end of namespace " + argv[len(argv) - i] + '\n')
        #if i < len(argv) - 1:
        #    f.write('\n')
    f.write('\n')
    f.write("#endif  //----end of " + define_name + "\n")   #like #endif //end of COMPRESSOR_H_
    f.close()
    command = "svn add %s" % (file_name)
    os.system(command)
    command = "win-path.py %s"%(file_name)
    os.system(command)
if __name__ == '__main__':
    run(sys.argv[1:])
