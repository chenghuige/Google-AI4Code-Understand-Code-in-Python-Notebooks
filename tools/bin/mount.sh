#sudo rm /home/users/chenghuige 
#sud mkdir -p /home/users/chenghuige/
#sudo mount //cq01-forum-rstree01.cq01.baidu.com/img /home/users/chenghuige -o username=root,password=run,uid=1000,gid=10 
#mount.cifs //yq01-image-gpu-1.yq01.baidu.com/img /home/users/chenghuige -o username=root,password=123456,uid=1000,gid=10
mount.cifs //yq01-image-gpu-1.yq01.baidu.com/img /home/gpu1 -o username=root,password=123456,uid=1000,gid=10
mount.cifs //yq01-image-gpu-3.yq01.baidu.com/img /home/gpu3 -o username=root,password=123456,uid=1000,gid=10
