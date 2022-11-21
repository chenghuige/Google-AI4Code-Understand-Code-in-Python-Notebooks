 /home/forum/lamp/mysql5/bin/mysql  -uroot -proot evaluation-pf -N -e "select  distinct post_id, max(dict_id) as dict_id,task_id,pv_id from  epf_post_dict where task_id > $1 and pv_id = 0 group by post_id having dict_id = 1  order by task_id desc" > thread.normal.pids 

