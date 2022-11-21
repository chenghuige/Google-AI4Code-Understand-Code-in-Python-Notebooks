#! /usr/bin/env python
#coding=utf-8
from sgmllib import SGMLParser
import urllib,re,sys,os

os.chdir('.')
collect_ = set()
f=open("url.txt","a")

class UrlList(SGMLParser):
    def reset(self):
        self.urls=[]
        SGMLParser.reset(self)
    def start_a(self,attrs):
        href=[v for k,v in attrs if k=='href']
        if href:
            self.urls.extend(href)
def getUrls(url):
    try:
        usock=urllib.urlopen(url)
    except:
        print "get url except"+url
        return []
    result=[]
    parser=UrlList()
    try:
      parser.feed(usock.read())
      usock.close()
      parser.close()
      urls=parser.urls
    except Exception:
      return []
    main_url = url[:url.rfind('/') + 1]
    for url in urls:
        #if len(re.findall(r'pdf',url))>0:  #指定正则表达式
        #if (url.endswith('pdf') or url.endswith('ps') or url.endswith('doc') or url.endswith('ppt')):
        #if (not url.endswith('html')):
        if not (url.startswith('http:') or url.startswith('ftp:')):
          if (url.startswith('./')):
            url = url[2:]
          url = main_url + url
        result.append(url)
    return result


def urlCanWrite(url):
  ends_notok = ['.html', '.edu', '.com', '.org']
  for str in ends_notok:
    if url.endswith(str):
      return False
  ends_ok =['.pdf','.ps','.doc','.ppt','.zip','.gz','.r','.mat','.xls']
  for str in ends_ok:
    if url.endswith(str):
      return True
  return False

def spider(startURL,depth):
    global collect_
    global f
    if depth<0:
        return
    else:
      collect_.add(startURL)
      urls=getUrls(startURL)
      for url in urls:
        #if (url.endswith('.html')):
        spider(url, depth - 1)
        can_write = urlCanWrite(url)
        if (can_write):
          collect_.add(url)
        
if __name__=="__main__":
    url = 'http://www.cs.cmu.edu/~epxing/Class/10701/lecture.html'
    if len(sys.argv) > 1:
      url = sys.argv[1]
    depth = 0
    if len(sys.argv) > 2:
      depth = 1
    spider(url,depth)  #指定需处理网页
    for url in collect_:    
      f.write(url+"\n")

