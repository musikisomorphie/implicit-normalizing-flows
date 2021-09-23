deepspeed --include=localhost:0,1 --master_port 50123 train_re.py --cuda  --deepspeed_config config_rxrx1.json \
    --dataset rxrx1 --dataroot /home/miashan/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 4 --scale-factor 1 \
    --env '201' --imagesize 256 --batchsize 32 \
    --nepochs 90 --nblocks 4-4-4-4 --factor-out --actnorm --symm-batchsize 8 \
    --task hybrid --nworkers 2 --eval-batchsize 32 --print-freq 120 --vis-freq 400 \
    --act elu --update-freq 5 --n-exact-terms 8 --fc-end False