#!/usr/bin/env python
import subprocess, time, sys, os 
import gezi

total = int(sys.argv[1])

interval = 1

class Superviser():
  def __init__(self):
    self.step = 0
    self.command = 'sh %s' % ' '.join(sys.argv[2:])
    self.run()
    while True:
      print(f'step:{self.step}, I will sleep {interval}s')
      time.sleep(interval)
      self.run()
  
  def run(self):
    gpus = gezi.get_gpus()
    print(f'step:{self.step}, start running:{self.command}')
    #os.system(f'CUDA_VISIBLE_DEVICES={gpus[0]} {command}')
    ret = subprocess.run(['env', f'CUDA_VISIBLE_DEVICES={gpus[0]}', *(self.command.split())]).returncode
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
