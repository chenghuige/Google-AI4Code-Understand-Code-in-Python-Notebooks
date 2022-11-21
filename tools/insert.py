import os  
import fileinput  
def file_insert(fname,linenos=[],strings=[]):  
    """ 
    Insert several strings to lines with linenos repectively. 
 
    The elements in linenos must be in increasing order and len(strings) 
    must be equal to or less than len(linenos). 
 
    The extra lines ( if len(linenos)> len(strings)) will be inserted 
    with blank line. 
    """  
    if os.path.exists(fname):  
        lineno = 0  
        i = 0  
        for line in fileinput.input(fname,inplace=1):  
            # inplace must be set to 1  
            # it will redirect stdout to the input file  
            lineno += 1  
            line = line.strip()  
            if i<len(linenos) and linenos[i]==lineno:  
                if i>=len(strings):  
                    print "\n",line  
                else:  
                    print strings[i]  
                    print line  
                i += 1  
            else:  
                print line  
file_insert('a.txt',[1,4,5],['insert1','insert4'])  
