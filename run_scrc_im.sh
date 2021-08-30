deepspeed --include=localhost:4,5,6,7 --master_port 54567 train_im.py --cuda  --data scrc --deepspeed_config config_im.json \
    --save 'experiments/scrc_zero_pad/' \
    --dataroot /raid/jiqing/Data/SCRC/ \
    --flow imflow --classifier resnet --scale-factor 4 --env '201' --aug 'rr'  \
    --inp 'im' --oup 'cms' --couple-label False --imagesize 128 --batchsize 16 \
    --actnorm True --task hybrid --nworkers 2 --val-batchsize 32 --nepochs 100 \
    --nblocks 9-9-9 --print-freq 120 --factor-out True --squeeze-first True \
    --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --fc False --coeff 0.9 --n-exact-terms 10

