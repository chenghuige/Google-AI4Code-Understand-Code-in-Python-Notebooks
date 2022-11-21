#!/usr/bin/env python
import subprocess, time, sys, os 
import gezi

class Superviser():
  def __init__(self):
    self.num_gpus = int(sys.argv[1])
    commands = sys.argv[2:]
    if commands[0] in ['sh', 'python']:
      command = ' '.join(commands)
    else:
      if commands[0].endswith('.py'):
        command = 'python %s' % ' '.join(commands)
      else:
        command = 'sh %s' % ' '.join(commands)
    self.command = command
    self.run()
  
  def run(self):
    command = self.command
    gpus = gezi.get_gpus()[:self.num_gpus]
    gpus = ','.join(gpus)
    print(f'start running:{self.command} with gpus:{gpus}')
    os.system(f'CUDA_VISIBLE_DEVICES={gpus} {self.command}')

worker = Superviser()
