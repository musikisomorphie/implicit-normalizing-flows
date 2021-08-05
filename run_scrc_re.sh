CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_re.py --data scrc --actnorm True --task hybrid \
    --nblocks 2-2-2 --classifer densenet \
    --save 'experiments/scrc_re' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 16 --nepochs 100
