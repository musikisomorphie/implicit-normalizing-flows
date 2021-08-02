CUDA_VISIBLE_DEVICES=0 python train_img_im.py --data scrc --actnorm True --task hybrid \
    --nblocks '1-1-1' --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --preact True \
    --save 'experiments/scrc(blocks_2*3(512,k313)_swish_nofc_preact_10term' --coeff 0.9 --n-exact-terms 10 --imagesize 64 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 64
