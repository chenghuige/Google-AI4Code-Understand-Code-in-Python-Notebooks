#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   b.py
#        \author   chenghuige  
#          \date   2012-01-03 16:12:01.789950
#   \Description   
# ==============================================================================

import sys,os
import pexpect  
import os  
 
user = 'chenghuige'  
host = 'relay01.baidu.com'
user2 = 'work'
host2 = 'bb-test-nlp09.vm'

password = ''
if (len(sys.argv) >= 2):
  password = '1229%s'%sys.argv[1]  
else:
  password = '1229'
  
password2 = 'testnlp#123'



foo = pexpect.spawn('ssh %s@%s'%(user, host))  
foo.expect('.CODE:*')
foo.sendline(password)   

#foo.expect('*$')
foo.expect(pexpect.EOF)
foo.sendline('ssh %s@%s'%(user2, host2))

foo.expect('*assword:*')
foo.send(password2)

foo.interact()  



 
