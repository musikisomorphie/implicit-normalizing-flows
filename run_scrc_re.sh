CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_re.py --cuda  --data scrc \
    --actnorm True --task hybrid --nworkers 2 \
    --nblocks 6-6-6 \
    --save 'experiments/scrc_re' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 16 --val-batchsize 32 --nepochs 100