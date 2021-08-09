CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train_img_re --data scrc --actnorm True --task hybrid \
    --nblocks 4-4-4 \
    --save 'experiments/scrc_re' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 16 --val-batchsize 32 --nepochs 100
