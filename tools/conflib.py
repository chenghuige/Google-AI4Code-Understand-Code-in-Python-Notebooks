#!/usr/bin/env python
#coding=utf-8
import traceback
import sys
import time
import ConfigParser

class ConfLib:
    def __init__(self, configFile, reload=False):
        """
        configFile: 配置文件
        reload: 读配置时每次都重载标识
        """
        self.configFile = configFile
        if not reload:
            self.config = ConfigParser.ConfigParser()
            self.config.read(self.configFile)
        self.reload = reload

    def __trace(self, msg=""):
        print "%s\t config.get() failed, and using default value:" % (time.strftime("%Y-%m-%d %H:%M:%S")),
        print msg
        excType, excVal, excTb = sys.exc_info()
        fmtExc = traceback.format_exception(excType, excVal, excTb, limit=1)
        print "    Traceback: %s    %s" % (fmtExc[1], fmtExc[-1])

    def __getconf(self, reload):
        if reload:
            config = ConfigParser.ConfigParser()
            config.read(self.configFile)
        else:
            config = self.config
        return config

    def __getFunc(self, section, option, defval, funcname):
        config = self.__getconf(self.reload)
        try:
            #retval = eval("config." + funcname)(section, option)
            retval = getattr(config, funcname)(section, option)
        except:
            retval = defval
            self.__trace(defval)
        return retval

    def get(self, section, option, defval=""):
        return self.__getFunc(section, option, defval, "get")
    
    def getint(self, section, option, defval=0):
        return self.__getFunc(section, option, defval, "getint")

    def getfloat(self, section, option, defval=0.0):
        return self.__getFunc(section, option, defval, "getfloat")

    def getboolean(self, section, option, defval=False):
        return self.__getFunc(section, option, defval, "getboolean")

    def items(self, section):
        config = self.__getconf(self.reload)
        try:
            retval = config.items(section)
        except:
            retval = None
            self.__trace()
        return retval

def __test():
    file = "test.conf"
    conf = ConfLib(file, True)
    str = conf.get("Section", "item", "apple")
    print type(str), str
    ints = conf.getint("Section", "int", 23)
    print type(ints), ints
    fs = conf.getfloat("Section", "float", 3.5)
    print type(fs), fs
    bs = conf.getboolean("Section", "boolean", True)
    print type(bs), bs
    print conf.items("Section")

if __name__ == "__main__":
    __test()

