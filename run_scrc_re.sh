CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_re.py --data scrc --actnorm True --task hybrid \
    --nblocks 1-1-1 \
    --save 'experiments/scrc_re' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 16 --val-batchsize 32 --nepochs 100
