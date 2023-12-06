python train.py --dataroot ./datasets/isbi24_final_v2 --name dapi_isbi24_v2 --dataset_mode unaligned_onehotseg_onecycle_isbi24 --model cycle_gan_noKL_one_cycle_onehotseg --input_nc 1 --output_nc 1 --batch_size 4 --gpu_ids 1 --num_threads 32 --load_size 1024 --crop_size 512 --patch_size 1024 --preprocess scale_width_and_crop