# similar to
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_re.py --data scrc --actnorm True --task hybrid \
    --nblocks 32-32-32 --factor-out True  --squeeze-first True \
    --save 'experiments/scrc_re' --imagesize 64 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 64 --nepochs 100
