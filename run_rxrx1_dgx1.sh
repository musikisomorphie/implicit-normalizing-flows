deepspeed --include=localhost:2,3 --master_port 54567 train_re.py --cuda  --deepspeed_config config_rxrx1.json \
    --dataset rxrx1 --dataroot /raid/jiqing/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 1 \
    --env '201' --imagesize 128 --batchsize 64 \
    --nepochs 90 --nblocks 8-8-8 --factor-out --actnorm \
    --task hybrid --nworkers 2 --eval-batchsize 64 --print-freq 120 --vis-freq 400
    