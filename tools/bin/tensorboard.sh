#/home/gezi/tensorflow.cpu/bin/tensorboard --logdir $1 --port $2
#python /home/gezi/other/tensorflow/tensorflow/tensorflow/tensorboard/tensorboard.py --logdir $1 --port $2 
#python /usr/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir $1 --port $2 
#CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir $1 --port $2  --bind_all
CUDA_VISIBLE_DEVICES=-1 tensorboard --logdir $1 --port $2  
