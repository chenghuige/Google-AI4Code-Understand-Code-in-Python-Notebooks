#!/usr/bin/env python
import subprocess, time, sys, os

interval = 1

class Superviser():
  def __init__(self):
    self.step = 0
    self.command = 'sh %s' % ' '.join(sys.argv[1:])
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
    print(f'step:{self.step}, start running:{self.command}')
    os.system(self.command)
    self.step += 1

worker = Superviser()
