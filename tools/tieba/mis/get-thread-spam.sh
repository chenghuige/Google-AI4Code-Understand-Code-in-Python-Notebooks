/home/forum/lamp/mysql5/bin/mysql  -uroot -proot evaluation-pf -N -e "select  distinct post_id, dict_id,task_id,pv_id from  epf_post_dict where task_id > $1 and pv_id =0 and not (dict_id = 1 or dict_id = 37 or dict_id = 0) order by task_id desc" > thread.spam.pids  

