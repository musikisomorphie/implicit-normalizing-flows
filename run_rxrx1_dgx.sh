deepspeed --include=localhost:0,1,2,3 --master_port 50123 train_re.py --cuda  --deepspeed_config config_rxrx1.json \
    --dataset rxrx1 --dataroot /raid/jiqing/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 0.25 \
    --env '201' --imagesize 256 --batchsize 72 \
    --nepochs 100 --nblocks 8-8-8 --factor-out --actnorm \
    --task hybrid --nworkers 2 --eval-batchsize 72 --print-freq 120 --vis-freq 400
    