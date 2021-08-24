deepspeed --include=localhost:4,5,6,7 --master_port 54567 train_im.py --cuda  --data scrc --deepspeed_config config_im.json \
    --actnorm True --task hybrid --nworkers 2 \
    --nblocks 12-12-12 --print-freq 120 --factor-out True --squeeze-first True \
    --save 'experiments/scrc_im_alpha/' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC/ --batchsize 16 --val-batchsize 32 --nepochs 100 \
    --env '201' --aug 'rr' --inp 'im' --scale-factor 4 --flow imflow \
    --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --fc False --coeff 0.9 --n-exact-terms 10 \

