from translator import *
import MySQLdb
from conflib import *

#getLevel(01.02.00.00.00.00) = 2
def getLevel(path):
    level = 0;
    for i in range(0, len(path), 2):
        if (path[i]!='0' or path[i+1]!='0'):
            level += 1
        else:
            break
    return level
        
li = []

def readDB(_ip, _db, _table, _user, _passwd, _out):
  conn = MySQLdb.connect(host=_ip, user=_user, passwd=_passwd, db=_db, charset='utf8')
  cur = conn.cursor()
  #sql = "SELECT category_id, category_path, category_name FROM " + _table
  sql = "SELECT * FROM "+_table
  cur.execute(sql)
  rows = cur.fetchall()

  digits_only = translator(keep=string.digits)

  for line in rows: 
    id, path, name = line[0], line[1], line[2] 
    path = digits_only(str(path))
    level = getLevel(path)
    name = '/'.join(name.split())
    li.append([str(id), path.encode('gbk'), name.encode('gbk'), str(level)])

  cur.close()
  li.sort(key = lambda x:x[1])  #sort by path

  for item in li:
    _out.write(' '.join(item))
    _out.write('\n')

def run(config_file):
  #read conf
  conf = ConfLib(config_file, True)
  IP = conf.get("DB_Category", "IP", "172.16.128.57")
  DB = conf.get("DB_Category", "DB", "ProductPage")
  TABLE = conf.get("DB_Category", "TABLE", "category")
  USER = conf.get("DB_SearchResult", "USER", "readuser")
  PASSWD = conf.get("DB_SearchResult", "PASSWD", "password") 
  out_file = conf.get("DB_Category", "OutCategoryInfo", "category_sorted.txt")
  out = open(out_file, 'w')

  readDB(IP,DB,TABLE,USER,PASSWD,out)

if __name__ == "__main__":
  config_file = "read_db.conf"
  if (len(sys.argv) > 1):
    config_file = sys.argv[1]
  run(config_file)  

