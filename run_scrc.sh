CUDA_VISIBLE_DEVICES=0 python train_scrc.py --data scrc --actnorm True --task hybrid \
    --nblocks 1-1-1 --print-freq 60 \
    --save 'experiments/' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 32 --val-batchsize 64 --nepochs 100 \
    --env '201' --aug 'r' --inp 'i'
