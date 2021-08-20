CUDA_VISIBLE_DEVICES=0,1,2,3 python train_im.py --cuda  --data scrc \
    --actnorm True --task hybrid --nworkers 2 \
    --nblocks 8-8-8 --print-freq 120 --factor-out True --squeeze-first True \
    --save 'experiments/scrc_im_alpha/' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC/ --batchsize 16 --val-batchsize 32 --nepochs 100 \
    --env '201' --aug 'rr' --inp 'i' \
    --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --coeff 0.9 --n-exact-terms 10 \

