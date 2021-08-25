deepspeed --include=localhost:0 --master_port 50123 train_im.py --cuda  --data scrc --deepspeed_config config_im.json \
    --save 'experiments/scrc/' \
    --dataroot /home/histopath/Data/SCRC_nuclei/ \
    --flow imflow --classifier resnet --scale-factor 2 --env '201' --aug 'rr'  \
    --inp 'im' --oup 'cms' --couple-label False --imagesize 128 --batchsize 2 \
    --actnorm True --task hybrid --nworkers 2 --val-batchsize 4 --nepochs 100 \
    --nblocks 2-2-2 --print-freq 120 --factor-out True --squeeze-first True \
    --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --fc False --coeff 0.9 --n-exact-terms 10 \
    
