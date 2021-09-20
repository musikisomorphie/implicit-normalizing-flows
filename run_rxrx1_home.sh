deepspeed --include=localhost:0,1 --master_port 50123 train_re.py --cuda  --deepspeed_config config_rxrx1.json \
    --dataset rxrx1 --dataroot /home/miashan/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 0.25 \
    --env '201' --imagesize 256 --batchsize 48 \
    --nepochs 100 --nblocks 2-2-2 --factor-out --actnorm \
    --task hybrid --nworkers 2 --eval-batchsize 24 --print-freq 120 --vis-freq 400
    