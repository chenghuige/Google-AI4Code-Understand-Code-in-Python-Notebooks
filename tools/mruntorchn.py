#!/usr/bin/env python
import subprocess, time, sys, os
import random
import gezi

total = int(sys.argv[1])
num_gpus = int(sys.argv[2])
port = sys.argv[3]

interval = 1

class Superviser():
  def __init__(self):
    self.step = 0
    self.command = ' '.join(sys.argv[4:])
    self.run()
    while True:
      print(f'step:{self.step}, I will sleep {interval}s')
      time.sleep(interval)
      self.run()
  
  def run(self):
    seed = random.randint(0, 100000)
    self.command += f' --seed={seed}'
    gpus = gezi.get_gpus()
    gpus = gpus[:num_gpus] if gpus else range(num_gpus)
    gpus_str = ','.join(map(str, gpus))

    # world size in terms of number of processes
    dist_world_size = num_gpus

    node_rank = 0
    # set PyTorch distributed related environmental variables
    env = os.environ.copy()
    current_env = {}
    current_env["MASTER_ADDR"] = '127.0.0.1'
    current_env["MASTER_PORT"] = port
    current_env["WORLD_SIZE"] = str(num_gpus)
    current_env["CUDA_VISIBLE_DEVICES"] = gpus_str

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and num_gpus > 1:
      current_env["OMP_NUM_THREADS"] = str(1)

    for local_rank in range(0, num_gpus):
      # each process's rank
      dist_rank = num_gpus * node_rank + local_rank
      current_env["RANK"] = str(dist_rank)
      current_env["LOCAL_RANK"] = str(local_rank)
      env.update(current_env)

      print(f'step:{self.step}, start running:{self.command} env:{current_env}')
      process = subprocess.Popen(self.command.split(), env=env)
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
