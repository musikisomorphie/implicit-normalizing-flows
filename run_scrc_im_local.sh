CUDA_VISIBLE_DEVICES=0,1 deepspeed train_img_im.py --cuda  --data scrc --deepspeed_config config_im.json \
    --actnorm True --task hybrid \
    --nblocks 1-1-1 --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --preact True \
    --save 'experiments/scrc_im' --imagesize 64 --coeff 0.9 --n-exact-terms 10 \
    --dataroot /home/miashan/Data/SCRC --batchsize 8 --val-batchsize 8 --nepochs 100