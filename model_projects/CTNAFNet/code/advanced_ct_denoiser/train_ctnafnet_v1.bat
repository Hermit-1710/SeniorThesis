@echo off
cd /d E:\a_ST\CTformer\CTformer-main
"E:\Anaconda\envs\CTformer\python.exe" .\main.py ^
  --model_name ctnafnet ^
  --loss_name hybrid ^
  --optimizer adamw ^
  --scheduler cosine ^
  --saved_path .\npy_img_3mm_B30 ^
  --save_path .\runs\ctnafnet_v1 ^
  --device cuda ^
  --batch_size 4 ^
  --patch_n 4 ^
  --patch_size 64 ^
  --num_epochs 201 ^
  --print_iters 20 ^
  --save_iters 500 ^
  --decay_iters 0 ^
  --lr 2e-4 ^
  --weight_decay 1e-4 ^
  --num_workers 0 ^
  --result_fig false
