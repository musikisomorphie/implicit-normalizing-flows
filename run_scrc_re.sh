CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_re.py --cuda  --data scrc \
    --actnorm True --task hybrid --nworkers 1 \
    --nblocks 4-4-4 \
    --save 'experiments/scrc_re' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 12 --val-batchsize 24 --nepochs 100