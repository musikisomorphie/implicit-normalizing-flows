deepspeed --include=localhost:4,5,6,7 --master_port 54567 train_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --save 'experiments/scrc_zero_pad_resize128/' \
    --dataroot /raid/jiqing/Data/SCRC/ \
    --flow reflow --classifier resnet --scale-factor 2 --env '201' --aug 'rr'  \
    --inp 'im' --oup 'cms' --couple-label False --imagesize 128 --batchsize 16 \
    --actnorm True --task hybrid --nworkers 2 --val-batchsize 32 --nepochs 100 \
    --nblocks 12-12-12 --print-freq 120 --factor-out True --squeeze-first True