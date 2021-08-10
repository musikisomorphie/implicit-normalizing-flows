CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_im.py --data scrc --actnorm True --task hybrid  --nworkers 2\
    --nblocks 4-4-4 --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --preact True \
    --save 'experiments/scrc_im' --imagesize 128 --coeff 0.9 --n-exact-terms 10 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 12 --val-batchsize 16 --nepochs 100
