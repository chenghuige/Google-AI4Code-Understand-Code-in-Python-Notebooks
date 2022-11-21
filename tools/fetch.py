#! /usr/bin/env python
#coding=utf-8
from sgmllib import SGMLParser
import urllib,re,sys,os

os.chdir('.')

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
    parser.feed(usock.read())
    usock.close()
    parser.close()
    urls=parser.urls
    main_url = url[:url.rfind('/') + 1]
    for url in urls:
        #if len(re.findall(r'pdf',url))>0:  #指定正则表达式
        #if (url.endswith('pdf') or url.endswith('ps') or url.endswith('doc') or url.endswith('ppt')):
        if (not url.endswith('html')):
          if not (url.startswith('http:') or url.startswith('ftp:')):
            if (url.startswith('./')):
              url = url[2:]
            url = main_url + url
          #print url
          result.append(url)
    return result

def spider(startURL,depth):
    f=open("url.txt","a")
    #if depth<0:
    #    return
    #else:
    urls=getUrls(startURL)
    for url in urls:
      f.write(url+"\n")
    
if __name__=="__main__":
    url = 'http://www.cs.cmu.edu/~epxing/Class/10701/lecture.html'
    if len(sys.argv) > 1:
      url = sys.argv[1]
    spider(url,0)  #指定需处理网页
