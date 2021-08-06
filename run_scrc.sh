CUDA_VISIBLE_DEVICES=0,1 python train_scrc.py --data scrc --actnorm True --task hybrid \
    --nblocks 1-1-1 --print-freq 40 \
    --save 'experiments/scrc' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 32 --nepochs 100
