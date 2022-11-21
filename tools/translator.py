# -*- coding: utf-8 -*-
import string
def translator(frm='', to='', delete='', keep=None): 
    if len(to) == 1:
        to = to * len(frm)
    trans = string.maketrans(frm, to)
    if keep is not None:
        allchars = string.maketrans('', '')
        delete = allchars.translate(allchars, keep.translate(allchars, delete)) 
    def translate(s):
        return s.translate(trans, delete)
    return translate

if __name__ == '__main__':
  digits_only = translator(keep=string.digits)
  print digits_only('Chris Perkins : 224-7992')
  #'2247992' 可以过滤指定的字符: 
  no_digits = translator(delete=string.digits)
  print no_digits('Chris Perkins : 224-7992') 
  #‘Chris Perkins : -’ 
  
  #可以替换指定的字符:
  digits_to_hash = translator(frm=string.digits, to='#')
  print digits_to_hash('Chris Perkins : 224-7992') 
  #‘Chris Perkins : ###-####’
