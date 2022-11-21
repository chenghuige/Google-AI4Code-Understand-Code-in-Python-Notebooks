# Google-AI4Code-Understand-Code-in-Python-Notebooks
4th solution  
# notice my path is /work/pikachu, be sure to set to your own path  
export PATH=/work/pikachu/tools:/work/pikachu/tools/bin:$PATH   
export PYTHONPATH=/work/pikachu/utils:/work/pikachu/third:$PYTHONPATH   
cd projects/kaggle/ai4code/  
# ls to see the link path of input and working, make sure you creat your own path and link it here  
cd src 
## pairwise model  
### mlm pretrain for mpnet  
sh scripts/mlm-prepare-mpnet.sh    
sh scripts/mlm3-mpnet.sh  
### pairwsie model using 9 negatives(notice you could set it to smaller number like 4 if you want to run faster)   
flag.sh flags/pairwise14-2-pre_mlm3 --num_negs=9 --online   
