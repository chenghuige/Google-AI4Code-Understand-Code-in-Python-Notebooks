#!/usr/bin/env python
import subprocess, time, sys, os

command = 'sh %s' % ' '.join(sys.argv[1:])

interval = 1

def get_command(command, is_first=True):
  l = command.split()
  if is_first:
    l += ['--rounds=1']
    return ' '.join(l)
  else:
    l2 = []
    for item in l:
      if 'restore_include' in item:
        continue
      if 'restore_exclude' in item:
        continue 
      if 'clear' in item:
        continue
      l2 += [item]
    return ' '.join(l2)

class Superviser():
  def __init__(self):
    self.step = 0
    self.run(get_command(command, is_first=True))
    while True:
      print(f'step:{self.step}, I will sleep {interval}s')
      time.sleep(interval)
      self.run(get_command(command, is_first=False))
  
  def run(self, command):
    gpus = gezi.get_gpus()
    print(f'step:{self.step}, start running:{command}')
    os.system(f'CUDA_VISIBLE_DEVICES={gpus[0]} {command}')
    self.step += 1

worker = Superviser()
