#!/usr/bin/env python
import subprocess, time, sys, os
import random
import gezi

total = 1
num_gpus = int(sys.argv[1])

interval = 1

class Superviser():
  def __init__(self):
    self.step = 0
    
    commands = sys.argv[2:]
    if commands[0].endswith('.py'):
      self.command = 'python %s' % ' '.join(commands)
    else:
      self.command = 'sh %s' % ' '.join(commands)

    self.run()
    while True:
      print(f'step:{self.step}, I will sleep {interval}s')
      time.sleep(interval)
      self.run()
  
  def run(self):
    seed = random.randint(0, 100000)
    self.command += f' --seed={seed} --num_gpus={num_gpus} --ps_strategy --job_name=worker'
    gpus = gezi.get_gpus()
    gpus = gpus[:num_gpus] if gpus else range(num_gpus)
    gpus_str = ','.join(map(str, gpus))

    # world size in terms of number of processes
    dist_world_size = num_gpus

    node_rank = 0
    env = os.environ.copy()
    current_env = {}
    current_env["WORLD_SIZE"] = str(num_gpus)
    # current_env["CUDA_VISIBLE_DEVICES"] = gpus_str

    processes = []
    command = self.command.replace('worker', 'ps')
    process = subprocess.Popen(command.split(), env=env)
    print(f'step:{self.step}, start running:{command} env:{current_env}')
    processes.append(process)

    for local_rank in range(0, num_gpus):
      # each process's rank
      dist_rank = num_gpus * node_rank + local_rank
      current_env["RANK"] = str(dist_rank)
      current_env["LOCAL_RANK"] = str(local_rank)
      env.update(current_env)
      command = self.command + f' --task_index={local_rank}'
      print(f'step:{self.step}, start running:{command} env:{current_env}')
      process = subprocess.Popen(command.split(), env=env)
      processes.append(process)

    ret = 0
    for process in processes:
      process.wait()
      if process.returncode != 0:
        ret = process.returncode
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=cmd)

    if ret == 0:
      self.step += 1
    if self.step > 0:
      command = self.command
      self.command = self.command.replace('--clear', '')
      if self.command != command:
        self.step -= 1
    if self.step == total:
      print(f'Done {total}')
      exit(0)

worker = Superviser()


