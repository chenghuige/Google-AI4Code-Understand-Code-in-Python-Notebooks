#!/usr/bin/env python
import subprocess, time, sys, os 
import gezi

interval = 1

class Superviser():
  def __init__(self):
    self.step = 0
    commands = sys.argv[1:]
    if commands[0] in ['sh', 'python']:
      command = ' '.join(commands)
    else:
      if commands[0].endswith('.py'):
        command = 'python %s' % ' '.join(commands)
      else:
        command = 'sh %s' % ' '.join(commands)
    self.command = command
    self.run()
    while True:
      print(f'step:{self.step}, I will sleep {interval}s')
      time.sleep(interval)
      self.run()
  
  def run(self):
    if self.step > 0:
      command = self.command
      self.command = self.command.replace('--clear', '')
      if self.command != command:
        self.step -= 1
    gpus = gezi.get_gpus()
    print(f'step:{self.step}, start running:{self.command}')
    os.system(f'CUDA_VISIBLE_DEVICES={gpus[0]} {self.command}')
    self.step += 1

worker = Superviser()
