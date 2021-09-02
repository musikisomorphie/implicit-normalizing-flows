deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 50123 train_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --save 'experiments/scrc_zero_pad_resize64_circle_all/' \
    --dataroot /raid/jiqing/Data/SCRC/ \
    --flow reflow --classifier resnet --scale-factor 2 --env '201' --aug 'rr'  \
    --inp 'im' --oup 'cms' --couple-label False --imagesize 64 --batchsize 32 \
    --actnorm True --task hybrid --nworkers 2 --val-batchsize 64 --nepochs 500 \
    --nblocks 32-32-32 --print-freq 120 --factor-out True --squeeze-first True