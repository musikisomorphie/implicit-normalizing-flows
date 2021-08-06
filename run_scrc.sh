CUDA_VISIBLE_DEVICES=0 python train_scrc.py --data scrc --actnorm True --task hybrid \
    --nblocks 1-1-1 \
    --save 'experiments/scrc' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 16 --val-batchsize 32 --nepochs 100
