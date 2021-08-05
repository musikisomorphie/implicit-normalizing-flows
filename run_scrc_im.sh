CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img_im.py --data scrc --actnorm True --task hybrid \
    --nblocks 16-16-16 --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --preact True \
    --save 'experiments/scrc_im' --coeff 0.9 --n-exact-terms 10 --imagesize 64 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 32 --nepochs 100
