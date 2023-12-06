# fully supervised (baseline)
python train.py --gpu_ids 1 --name isbi24_supervised_v3_512p_vgg --no_instance --loadSize 512 --netG global --dataroot ./datasets/isbi24_supervised_v3/ --input_nc 3 --output_nc 3 --norm batch --batchSize 16 --label_nc 0
python train.py --gpu_ids 1 --name isbi24_supervised_v3_1024p_vgg --no_instance --netG local --ngf 32 --num_D 3 --load_pretrain checkpoints/isbi24_supervised_v3_512p_vgg --niter 50  --niter_decay 50 --niter_fix_global 10 --resize_or_crop none --norm batch --batchSize 4 --dataroot ./datasets/isbi24_supervised_v3/ --input_nc 3 --output_nc 3  --label_nc 0

# semi supervised (proposed)
python train.py --gpu_ids 1 --name isbi24_semi_supervised_v3_512p_vgg --no_instance --loadSize 512 --netG global --dataroot ./datasets/isbi24_semi_supervised_v3/ --input_nc 3 --output_nc 3 --norm batch --batchSize 16 --label_nc 0
python train.py --gpu_ids 1 --name isbi24_semi_supervised_v3_1024p_vgg --no_instance --netG local --ngf 32 --num_D 3 --load_pretrain checkpoints/isbi24_semi_supervised_v3_512p_vgg --niter 50  --niter_decay 50 --niter_fix_global 10 --resize_or_crop none --norm batch --batchSize 4 --dataroot ./datasets/isbi24_semi_supervised_v3/ --input_nc 3 --output_nc 3  --label_nc 0

