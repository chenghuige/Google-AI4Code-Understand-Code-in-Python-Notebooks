#!/usr/bin/env python
# -*- coding: utf-8 -*-
#file    h2cc.py
#author  pku_goldenlock@qq.com
#date    2009-10-29

#.h convert to .cpp framework
#注意!!
#1. 我们假定你的.h代码书写正确 当前不做异常，错误处理
#2. 函数的返回类型如 int 与 与函数名称要写到一行,否则会有问题,因为不太好判断函数名称.
#3. 如果已经存在要生成的文件,会只写入你在.h中新加入的函数,
#   但是注意原来已经存在的实现文件也应该是用h2cc.py生成的,因为我不是按语意判断,只是按照格式判断
#   新写的实现函数框架,会附加到原来的生成文件最后面,需要自己调整到需要的位置,尤其是有namespace的时候
# @TODO 注意当前要求namespace 和 {必须在同一行 
#   TODO  新写的函数直接放到原来生成文件中适当的位置
#4. 在生成的实现文件中,可能会把返回值单独一行,取决于行的长度,过长则分行
#5. 如果template模板写成多行,确保最后一行只有一个单独的>
#   template <
#     typename T = char,
#     template<typename> class _Decoder = HuffDecoder<int> #pattern like this might cause error 
#   >


#在某行发现) ; 之后,删除该行;之后的内容,该行开始的空白去掉
#改为 \n\{\n\}
# virtual = 0 对于纯虚函数也生成实体,如不需要自己去掉.  TODO 用户选项
#TODO 允许向SGI STL源代码那样,将返回值单独一行,在函数名称上面,不过似乎比较困难,因为当前
#我的程序基本按照(){}；template class typdef define // /* */识别这些字符的,仅仅考虑格式
#不考虑语意如果返回值
#单独一行,如何判断什么样的是返回值类型呢?比较麻烦.比如还要区分仅仅挨着占两行的两个函数声明.

#最常用的pattern
#查找如下 )  ;   或者 ) const ;  但是不包括纯虚函数
#()z中如找到     ;  //从队尾删除
#开始的空白要找到并删除，
#pattern = re.compile(r"([\s]*?).*[)](?!\s*=\s*0).*(;.*)\n$")

#class A : public 读出A未解决  done
#template                      done 支持模板类,模板函数,带有默认参数的模板,模板类中的模板函数,特化和偏特化
#                                   TODO  更复杂的情况可能有问题,比如模板类中嵌套模板类
#vitrual = 0                   done
#支持 const                    done
#宏   DFDFDFEE（ABC);          done
#把注释拷贝                    done
#已经有.cpp了  仅仅加新的接口的实现 partly done  
#忽略返回类型，返回类型不能重载
#operator   done

#TODO 
# class U {
# int uv() throw();
# }    
#当前会转换为
#int U::uv() U::throw() 
#{
#
#}

import sys,re,getopt,os

def usage():
    """ 
        Run the program like this
        
        ./h2cc.py abc.h ok.cc
        It will read abc.h and append the fuctions 
        to be implemented to ok.cc
        
        or 
           ./h2cc.py abc.h         #will create abc.cc
           ./h2cc.py abc.h cpp     #will create abc.cpp
        """


use_template = False
use_auto = False
silent = False
pattern_comment = re.compile(r'^\s*//')

#传入一个打开的文件(我们要生成的或者已经存在要分析处理)的handle，如abc.cc
#调用该函数者负责关闭该文件,该函数只负责处理
func_list_exist = []
outfile_exist_content = []
end_namespace_index = -1
def AnalyzeOutputFile(file_handle):
    global pattern_comment
    global func_list_exist
    global end_namespace_index
    global outfile_exist_content 
    print('Analyze output file right now ,the reslt of existing functions is below\n')
    file_handle.seek(0,0)
    m = file_handle.readlines()
    outfile_exist_content = m
    i = 0
    #---------------------------------------------------逐行处理
    while (i < len(m)):
        line = m[i][:m.find('//')]
        #print('yun ' + line)
        if end_namespace_index == -1 and re.search('end of namespace',line):
            #print('find!')
            end_namespace_index = i
        #---------------------判断是注释则略过  re.search(r'^s*$',line) 空行判断
        if re.search(r'^s*$',line) or pattern_comment.search(line): #null line or comment using //
            i += 1
            continue
        if re.search('^\s*/[*]', line):                             #comment using /*  
            while (not re.search('[*]/\s*$',line)):                 # */
                i += 1
                line = m[i]
            i += 1
            continue
        #-------------------------find exists functions
        find1 = re.search('[(]',line) 
        find2 = re.search('[)]',line)
        find3 = re.search('template',line)
        find4 = re.search('::',line)
        start_i = i
        if ((find1 or find3 or find4) and (not find2)):
            i += 1
            line += m[i] 
            while (i < len(m) and (not re.search('[)]',line))):
                i += 1
                line += m[i]
        if (i + 1 < len(m)):      
            find5 = re.search(r'^{\s*$', m[i + 1])  #非常重要的判断^{
            if not find5:
                i += 1
                continue
        #返回值单独一行的情况
        if re.search(r'^\s*(inline)?\s*[a-zA-Z0-9_]+\s*$', m[start_i - 1]):
            line = m[start_i - 1] + line
        match = re.search('^.*\)',line,re.MULTILINE|re.DOTALL)
        if match:
            print(match.group())
            func_list_exist.append(match.group())
        i += 1
    print('Output file analze done!')
    print('There are already %d functions exit\n'%len(func_list_exist))

#转化核心函数,当前比较乱,代码重复,一个循环中代码太长了 TODO 
#基本实现就是按照栈处理名称,区分是否class域的函数,忽略.h中已经实现函数{}内部的内容
def h2cc(input_file,output_file):
    """
        kernal function given a .h file
        convert to a .cc one with
        all the functions properly listed
        """
    print('The file you want to deal with is '+input_file + \
        '\n It is converted to ' + output_file)
    global pattern_comment
    global use_template
    global use_auto
    global end_namespace_index
    global func_list_exist
    global outfile_exist_content
    #----核心的函数匹配模式
    pattern = re.compile(r"""(^[\s]*)             #leading withe space,we will find and delete after
    		                 ([a-zA-Z~_]            # void int likely the first caracter v or i...
    						.* 
    						[)]                   #we find )
    						#(?!\s*=\s*0)          #if we find = 0  which means pur virtual we will not match after
                            #(?=\s*=\s*0) 
    						(?!.*{)              # we do not want the case int abc() const { return 1;}
                            .*)
    						(;.*)                 #we want to find ; and after for we will replace these later
    						\n$
    						""",re.VERBOSE | re.MULTILINE | re.DOTALL)
    
    #----处理virtual,explicit,friend,static 
    pattern2 = re.compile(r'(virtual\s+|explicit\s+|friend\s+|static\s+)')   
    
    #----开头的空白
    leading_space_pattern = re.compile('(^[\s]*)[a-zA-Z~_]')   
    
    #我们默认函数都会有 如 abc(  abc ( 这样的模式存在
    #但是operator 是个例外,类名要加在operaotr前面，而且不存在上面的模式
    #operator = ()     ClassName::operator = ()
    #pattern_func_name = re.compile(r'([a-zA-Z0-9~_\-]+\s*[(]|operator.*[(])')   
    #难道替换不是仅仅替换括号里面的 而是全部替换? 恩，大括号 必须的 然后用\1没有\0 
    pattern_func_name = re.compile(r'([a-zA-Z0-9~_\-]+\s*|operator.*)[(]') 

    pattern_template = re.compile('^\s*template')
    #pattern_template_end = re.compile('^\s*>\s*$') #TODO why wrong?
    pattern_template_end = re.compile('>\s*$')

    pattern_namespace = re.compile(r'namespace.*{')       #判断该行是否是 namespace出现
    #p2 = re.compile(r'class\s*(.*?)\s*{|struct\s*(.*?)\s*{')  
    #.*? 最小匹配  是否class出现,并记录class 名称
    pattern_class = re.compile(r'^[\s]*(class|struct)\s+([a-zA-Z0-9_\-]+<?)(?!.*;)') 
    #modify 09.6.6 可以处理classa a 和 { 不在同一行，但是如果class 后发现;不处理
    #class一定是行开始或者前面可有空白

    pattern_start = re.compile('{')
    pattern_end = re.compile('}')
    
    stack = []                      #----状态可能是normal_now(位于{}中间的时候),class_now,namespace_now
    stack_class = []                #----存储class name
    stack_template = []             #----存储template name
    stack_typedef = []              #----存储当前class 作用域下的所有typedef得到的名称,函数返回类型需要
    
    first_convert = True            #是否是第一次生成的实现文件
    
    #--------------------------------------文件处理
    try:                              #09.6.6 处理要转换生成的.cc 或 .cpp文件不存在的情况，添加读异常
        f_out = open(output_file,'r')    #r+  可读写但要求文件已经存在
        print(output_file + ' exists already, we will anlyze it to find the existing functions to avoid adding twice')
        AnalyzeOutputFile(f_out)                         #对输出文件进行预处理分析
        f_out.close()
        f_out = open(output_file,'w')
        if len(outfile_exist_content) > 0:
            for x in outfile_exist_content[:end_namespace_index]:
                f_out.write(x)
        first_convert = False
    except IOError:
        print(output_file + ' does not exist yet, we will create am empty ' + output_file)
        f_out = open(output_file,'a')     #追加打开，这里因为用了异常先判断了是否文件存在，所以也可以用 w
                                          #如果没有前面的判断只能用 a ,w 会自动删除前面已有的内容
    print('Below functions will be added\n')
    out_define = output_file.replace('.','_').upper() +'_' 
    if first_convert:
        if use_template:
            f_out.write("#define "+out_define+'\n')
        f_out.write('#include "' + input_file + '"\n\n')  #注意 如果out文件已存在 那么经过前面的分析处理 指针已经指向了文件尾部
     
    func_sum = 0
    namespace_num = 0
    write_define = 0
    #--------------------------------------------------核心处理循环,逐行处理输入.h文件
    with open(input_file,'r') as f:
        f2 = open(input_file+'.temp','w')
        m = f.readlines()
        i = 0
        while i < len(m):
            #m[i] = m[i][:m[i].find('//')]
            line = m[i]
            #print line
            #-------------------------------------------判断是注释则略过  re.search(r'^s*$',line) 空行判断
            if re.search(r'^s*$',line) or pattern_comment.search(line): #/n or comment using //
                f2.write(m[i])
                i += 1
                continue
            if re.search('^\s*/[*]', line):              #comment using /*  
                while (not re.search('[*]/\s*$',line)):  # */
                    f2.write(m[i])
                    i += 1
                    line = m[i]
                f2.write(m[i])
                i += 1
                continue
            #---------------------------------------------判断是则define略过
            define_match = re.search(r'^\s*#define',line)
            if define_match:
                while re.search(r'^\s*$',line) or re.search(r'\\\s*$', line):
                    f2.write(m[i])
                    i += 1
                    line = m[i]
                f2.write(m[i])
                i += 1
                continue
            #-----------------------------------------------判断是否namespace
            match_namespace = pattern_namespace.search(line)
            if match_namespace:                                   #we face namespace
                if first_convert:
                  f_out.write(line+'\n')
                stack.append('namespace_now')
                namespace_num += 1
                f2.write(m[i])
                i += 1
                continue 
            else:
                if i + 1 < len(m):
                    if line.lstrip().startswith('namespace') and m[i + 1].lstrip().startswith('{'):
                        if first_convert:
                            f_out.write(m[i])
                            f_out.write(m[i+1])
                        stack.append('namespace_now')
                        namespace_num += 1 
                        f2.write(m[i])
                        f2.write(m[i + 1])
                        i += 2
                        continue

            #----------------------------------------------------判断并处理类里面的typedef
            if (len(stack) > 0 and stack[-1] == 'class_now'):
                pattern_typedef = re.compile(r'typedef\s+.*\s+(.*);')
                match_typedef =  pattern_typedef.search(line)
                if match_typedef:
                    stack_typedef.append(match_typedef.group(1))
            #----------------------------------------------------判断并处理模板情况  
            match_template = pattern_template.search(line)
            template_string = ''
            if match_template:
                match_template_end = pattern_template_end.search(line)
                template_string = line
                while(not match_template_end):
                    f2.write(m[i])
                    i += 1
                    line = m[i]
                    template_string += line
                    match_template_end = pattern_template_end.search(line)
                f2.write(m[i])
                i += 1
                line = m[i]
            #--------------------------------------------判断是否是class 或者遇到 { start
            match_class = pattern_class.search(line)  
            match_start = pattern_start.search(line)

            if match_class:                  #we face a class
                stack_template.append(template_string)
                stack.append('class_now')
                class_name = match_class.group(2)   #TODO f2.group(1)如果为空则异常
                #-----------模板类特化或者偏特化的情况 如 class A<u,Node<u> > 为了获得整个名称
                if '<' in class_name:               
                    k = line.index('<')
                    fit = 1;
                    for l in range(k+1, len(line)):
                        if line[l] == '<':
                            fit += 1
                        if line[l] == '>':
                            fit -= 1
                        if (fit == 0):
                            break
                    class_name += line[k+1:l+1]
                stack_class.append(class_name)
                while not match_start:
                    f2.write(m[i])
                    i += 1
                    line = m[i]
                    match_start = pattern_start.search(line)
                f2.write(m[i])
                i += 1
                continue
            #-------------------------------------------------判断是否是结束符号 }
            match_end = pattern_end.search(line)
            if match_start:
                stack.append('normal_now')
            if match_end:
                top_status = stack.pop()
                if top_status == 'namespace_now':
                    if first_convert:
                        f_out.write(line+'\n') 
                    namespace_num -= 1
                elif top_status == 'class_now':
                    stack_class.pop()
                    stack_template.pop()
                    stack_typedef = []
            if match_start or match_end:
                f2.write(m[i])   #already done in if match_end
                if match_end and namespace_num == 0 and use_template and first_convert:
                    f2.write("#ifndef "+out_define+'\n')
                    f2.write('#include "'+output_file+'"\n')
                    f2.write("#endif\n")
                    write_define = 1
                i += 1
                continue
            #注意我判断是函数只是根据 我看到该行有) 然后 后面有;  important!!
            #------------------------------------------------就像忽略注释一样忽略normal_now状态下的行,因为那是在{}中间的实现
            if len(stack) >0 and stack[-1] == 'normal_now': 
                f2.write(m[i])
                i += 1
                continue
            #---------------------------------------------------------下面我们该处理需要生成实体框架的函数了,
            #deal with
            #int abc(int a,
            # 		 int b)    #能够处理这种(与)不在同一行的情况
            find1 = re.search('[(]',line)
            if not find1:
                f2.write(m[i])
                i += 1
                continue
            find2 = re.search('[)]',line)
            start_i = i
            space_match = leading_space_pattern.search(line)
            if (find1 and (not find2)):
                i += 1
                line2 = m[i]
                if space_match:
                    line2 = re.sub('^'+space_match.group(1),'',line2)     
                    #注意sub会替换所有的匹配，这里我们只让它替换第一个,count=1，或者对于space 前面加^
                line += line2
                while (i < len(m) and (not re.search('[)]',line2))):
                    i += 1
                    line2 = m[i]
                    line2 = re.sub('^'+space_match.group(1),'',line2)
                    line += line2

            match_start = pattern_start.search(m[i])
            match_end = pattern_end.search(m[i])
            if (match_start):     # like  ) {  or ) {}    int the last line
              if not match_end:
                stack.append('normal_now')
              ii = start_i                #fixed 09.11.17
              while (ii <= i):
                f2.write(m[ii])
                ii += 1
              i += 1
              continue
 
            #here we do the kernel sub  #--------------------------------如果找到,先进行了替换abc();->abc(){}
            #(line,match) = pattern.subn(r'\2 \n{\n\n}\n\n',line)  
            no_mark = 0
            func_line_temp = line
            if use_auto and (not re.search(';\s*$', line)):    #默认情况下将加上;使得它可以被转移到实现文件中
                line = line.rstrip()
                line += ';\n'
                no_mark = 1
                func_line_temp = line
            if (no_mark) and (not re.search(r'^\s*{\s*$', m[i+1])): 
                ii = start_i   #Temp modified is it ok?
                while (ii <= i):
                  f2.write(m[ii])
                  ii += 1
                i += 1
                continue
            (line,match) = pattern.subn(r'\2\n',line)  #key sub!!!
            #print '[' + line + ']' + '(' +  str(match) + ')'
            #temp add 尝试兼容返回值在单独一行的情况
            if re.search(r'^\s*(inline)?\s*[a-zA-Z0-9_]+\s*$', m[start_i - 1]):
                line = m[start_i - 1] + line
            line = line.lstrip() 
            #match = 1
            if (not match):   
                f2.write(m[i])
                i += 1
                continue
           
            #-------------------------------------------------------------OK,找到了函数,下面进行处理后输出
            friend_match = re.search('friend ',line)
            line = pattern2.sub('',line)            #--------------------delete virtural explict friend!
            func_name = ''
            template_line = ''
            if len(stack_class) > 0 and not friend_match :  #-----类成员函数class status if friend we will not add class name
                x = ''
                if (template_string != ''):                 #-----类中模板函数,多一个模板
                    template_string = re.sub(r'\s*template','template',template_string)
                    #template_string = re.sub(r'\s*=\s*[\'a-zA-Z0-9_\-\.]+', '', template_string)
                    #------------------delete < class T = a, class U = A(3)> -> <class T, class U>
                    template_string = re.sub('\s*=.*>(\s*)$',r'>\1',template_string)  
                    template_string = re.sub(r'\s*=.*,',',',template_string)
                    template_string = re.sub(r'\s*=.*','',template_string)
                if (stack_template[-1] != ''):
                    if not (re.search(r'<\s*>', stack_template[-1])): #----不是全特化!template<>
                        template_line = re.sub(r'^\s*template','template',stack_template[-1])
                        if not (re.search(r'<.*>', stack_class[-1])): #----不是偏特化!,like template<typename T> class A<T,int>
                            #------for x we get like template<class T, typename U> -> <T,U>
                            x = re.sub(r'template\s*<','<',template_line)    #remove template -> <class T, typename U>
                            x = re.sub(r'\n','',x)                           #取消分行,合并成一行
                            #去掉 = 及其后面的东西
                            x = re.sub(r'\s*=.*,', ',', x)
                            x = re.sub(r'\s*=.*\>', '>', x)
                            x = x.rstrip()             #remove \n
                            x = re.sub(r'(class|typename)\s+|(<class>|<typename>\s*class)','',x) #去掉class,typename ->  <T, U>
                            x = re.sub(r'<\s+','<',x)
                            x = re.sub(r'\s+>','>',x)
                            x = re.sub(r'\s+,', ',',x)
                            x = re.sub(r',\s+', ', ',x)  
                line = re.sub(r'\s*=\s*0','',line)     #纯虚函数我们也给函数实现体，这里特别判断一下去掉 = 0
                line = re.sub(r'\s*=.*,',',', line)  #去掉=后面的东西 foo(int a = cd(32));foo(int x =(3), int y= 4);
                line = re.sub(r'\s*=.*\)',')', line)   #先去掉所有的(),最后去掉()),这两个步骤都需要不能颠倒
                #---------如果函数较长,在void ABC::foo()  断开成两行 void ABC::\n foo()
                temp_line = pattern_func_name.sub(stack_class[-1] + x + '::' + r'\1(',line)
                if len(temp_line) > 60:
                    line = pattern_func_name.sub(stack_class[-1] + x + '::\n' + r'\1(',line)
                else:
                    line = temp_line
                #----------得到返回变量的类型 
                return_type = re.search('^(\S+)', line).group(1)
                if (return_type in stack_typedef):  #------对于返回值,需要指明是在该类中typedef指定的名称的情况
                    if (x == ''): #----全特化的情况特殊处理 
                        line = re.sub('^'+return_type+r'\s+', stack_class[-1]+'::'+return_type+'\\n', line)
                    else:
                        line = re.sub('^'+return_type+r'\s+', 'typename '+stack_class[-1]+'::'+return_type+'\\n', line)
                #----------add template as the above if there is one
                #template_line = re.sub(r'\s*=\s*[\'a-zA-Z0-9_\-\.]+', '', template_line)
                #------------------delete < class T = a, class U = A(3)> -> <class T, class U>
                template_line = re.sub('\s*=.*>(\s*)$',r'>\1',template_line)
                template_line = re.sub(r'\s*=.*,',',',template_line)
                template_line = re.sub(r'\s*=.*','',template_line) #最后,不换行删除后面所有的
                line = template_line + template_string +  line;        
                func_name = re.search('^.*\)',line,re.MULTILINE|re.DOTALL).group()
            else:                  #--------------------------------普通函数(非类成员函数)的情况!
                stack_template.append(template_string)
                if  (stack_template[-1] != ''):
                    template_line = re.sub(r'\s*template','template',stack_template[-1])
                    #template_line = re.sub(r'\s*=\s*[\'a-zA-Z0-9_\-\.]+', '', template_line)
                    #------------------delete < class T = a, class U = A(3)> -> <class T, class U>
                    template_line = re.sub('\s*=.*>(\s*)$',r'>\1',template_line) #代码重复,TODO以后整合 
                    template_line = re.sub(r'\s*=.*,',',',template_line)
                    template_line = re.sub(r'\s*=.*','',template_line)
                line = re.sub(r'\s*=.*,', ',', line)
                line = re.sub(r'\s*=.*\)', ')', line)
                line = template_line + line
                stack_template.pop()
                func_name = re.search('^.*\)',line,re.MULTILINE|re.DOTALL).group()
            #--------------------------------------------------------把已经在头文件定义的代码也拷贝过去
            content = ''
            lmatch = 0
            #特殊的写法对于{单独一行的情况把其上函数在头文件定义的代码也拷贝过去
            if i < len(m) - 1 and re.search(r'^\s*{\s*$', m[i+1]):               
                i = i + 2
                lmatch = 1
                while (lmatch != 0):
                    if (not pattern_comment.search(m[i])) and re.search('{', m[i]): #唯一可能的问题是注释 //  if (n > 1) {i
                        lmatch += 1
                    if (not pattern_comment.search(m[i])) and re.search(r'}',m[i]):
                        lmatch -= 1
                    if space_match:
                        content += re.sub('^'+space_match.group(1),'',m[i])
                    i += 1
                i -= 1
            if content == '':
                line += '{\n\n' + content +'}\n\n'
            else:
                line += '{\n' + content +'\n\n'
						#-------------------------------------------------------------------------加上上面的注释也拷贝过去
            comment_line = ''                          
            #假定函数的注释在函数声明的紧上面，把他们拷贝过去
            #TODO 如果是模板函数,模板占多于1行注释没拷贝
            k = start_i - 1    #one line befor this func start
            if pattern_template.search(m[k]):
                k -= 1
            if re.search('[*]/\s*$',m[k]):
                comment_line = m[k].lstrip()
                while not re.search('^\s*/[*]',m[k]):
                    k -= 1
                    comment_line = m[k].lstrip() + comment_line 
            else:    
              for j in range(k,0,-1):
                  c_line = m[j]
                  if pattern_comment.search(c_line):
                      c_line = re.sub('\s*//','//',c_line)  #TODO use strtip is ok?
                      comment_line = c_line + comment_line
                  else:
                      break
            line = comment_line + line  #------------------我们最终要输出的东东
            #----------------------------------------------如果函数已经在实现文件中存在,不输出
            if not(func_name in func_list_exist):
                #if not first_convert and func_sum == 0:
                #    f_out.write("//newly added functions, you may need to move them to the right position\n")
                f_out.write(line)
                func_sum += 1
                print(func_name)
            f2.write(func_line_temp)
            i += 1  #-----------------------------------------------------------------------next line处理下一行
    #------------------------------loop done 处理结束 
    if len(outfile_exist_content) > 0:
        print('outfile_exist_content is not empty')
        for x in outfile_exist_content[end_namespace_index:]:
            #print(x)
            f_out.write(x)
    #print("hahahaha %d"%use_template)
    if not write_define and (use_template and first_convert):           #注意你需要把最后那个文件名对应的endif 前面namespace{ }外面
        f2.write("#ifndef "+out_define+'\n')
        f2.write('#include "'+output_file+'"\n')
        f2.write("#endif\n")
        write_define = 1

    f2.close()
    command = 'cp '+input_file+' '+input_file+'.bak'        #save current input_file as input_file.bak,ie a.h ->a.h.bak
    os.system(command)  
    command = 'mv '+input_file+'.temp'+' '+input_file       #a.h.temp->a.h        
    os.system(command)
    print('\nSucessfully converted,please see '+output_file)
    print('Added %d functions'%func_sum)


#-----------------------------------------------------------user input
def main(argv):        
    global use_template
    global use_auto
    try:                                
        opts, args = getopt.getopt(argv, "htas", ["help","template","auto","silent"]) 
    except getopt.GetoptError:           
        print(usage.__doc__) 
        sys.exit(2)

    if len(opts) > 0:
        for o, a in opts: 
            if o in ("-h", "--help"): 
                print(usage.__doc__) 
                sys.exit()
            if o in ("-t", "--template"):
                use_template = True
            if o in ("-a", "--auto"):
                use_auto = True
            if o in ("-s", "--silent"):
                silent = True
    if len(args) > 0:
        input_file = args[0]
        if len(args) == 1 or args[1] == 'cc':
            output_file = re.sub('.h*$','.cc',input_file)  # 默认转的是对应的.cc文件
        elif (args[1] == 'cpp'):
            output_file = re.sub('.h*$','.cpp',input_file)
        else:
            output_file = args[1]
            
        h2cc(input_file,output_file)
    else:
        print(usage.__doc__)
        sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])
