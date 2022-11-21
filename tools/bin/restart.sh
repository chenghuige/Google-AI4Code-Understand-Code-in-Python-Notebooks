umount -l /search/odin/publicData/ 
cd /opt/hdfs-mount-diablo/; sh start.sh
smbd restart
cd /home/gezi/mine/pikachu/projects/feed/rank/src
nohup nc jup --ip=10.141.202.84 > /tmp/jup.log & 

