@echo off
set TEST_ITERS=%1
if "%TEST_ITERS%"=="" set TEST_ITERS=11500

cd /d E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer\code
"E:\Anaconda\envs\CTformer\python.exe" .\main.py ^
  --mode test ^
  --model_name ctrestormer ^
  --loss_name hybrid ^
  --optimizer adamw ^
  --scheduler cosine ^
  --saved_path E:\a_ST\CTformer\CTformer-main\npy_img_3mm_B30 ^
  --save_path E:\a_ST\CTformer\CTformer-main\model_projects\CTRestormer\runs\ctrestormer_v1 ^
  --device cuda ^
  --batch_size 2 ^
  --patch_n 4 ^
  --patch_size 64 ^
  --num_workers 0 ^
  --result_fig true ^
  --test_iters %TEST_ITERS% ^
  --lr 1e-4 ^
  --weight_decay 1e-4 ^
  --augment_mode advanced ^
  --restormer_dim 24 ^
  --restormer_num_blocks 1,2,2,3 ^
  --restormer_num_refinement_blocks 1 ^
  --restormer_heads 1,2,4,4 ^
  --grad_accum_steps 2 ^
  --use_amp true
