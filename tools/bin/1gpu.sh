gpu=`get-gpu.py`
CUDA_VISIBLE_DEVICES=$gpu $*
