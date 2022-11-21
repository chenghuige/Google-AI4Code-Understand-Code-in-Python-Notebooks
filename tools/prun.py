#!/usr/bin/env python
import subprocess, time, sys, os
import random
import gezi

if sys.argv[1].isdigit():
  parts = int(sys.argv[1])
  commands = sys.argv[2:]
else:
  parts = int(sys.argv[-1])
  commands = sys.argv[1:-1]

seed = random.randint(0, 100000)
commands = list(commands) + [f'--parts={parts} ', f'--seed={seed}']

if commands[0] in ['sh', 'python']:
  command = ' '.join(commands)
else:
  if commands[0].endswith('.py'):
    command = 'python %s' % ' '.join(commands)
  else:
    command = 'sh %s' % ' '.join(commands)

processes = []
env = os.environ.copy()

for part in range(parts):
  command_ = command
  command_ += f' --part={part}'
  print(command_)
  process = subprocess.Popen(command_.split(), env=env)
  processes.append(process)

ret = 0
for process in processes:
  process.wait()
  if process.returncode != 0:
    ret = process.returncode
    raise subprocess.CalledProcessError(returncode=process.returncode,
                                        cmd=cmd)
if ret == 0:
  if 'END' in os.environ:
    command = os.environ['END']
    os.system(command)
    
