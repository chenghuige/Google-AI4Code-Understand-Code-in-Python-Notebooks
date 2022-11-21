/home/forum/lamp/mysql5/bin/mysql  -uroot -proot evaluation-pf -N -e "select distinct post_id,dict_id,epf_post_dict.task_id,pv_id,task_type from epf_post_dict inner join epf_task on epf_post_dict.task_id = epf_task.task_id where epf_post_dict.task_id > $1 and task_type = 1 and not (dict_id = 1 or dict_id = 37 or dict_id = 0) order by task_id desc" > pv.spam.pids.reply

