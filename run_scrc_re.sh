deepspeed --include=localhost:0,1,2,3 --master_port 50123 train_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --save 'experiments/scrc/' \
    --dataroot /raid/jiqing/Data/SCRC/ \
    --flow reflow --classifier resnet --shuffle-factor 2 --env '201' --aug 'rr'  \
    --inp 'i' --couple-label False --imagesize 256 --batchsize 32 \
    --actnorm True --task hybrid --nworkers 2 --eval-batchsize 32 --nepochs 100 \
    --nblocks 12-12-12 --print-freq 120 --factor-out True --squeeze-first True