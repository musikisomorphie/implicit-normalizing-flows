deepspeed --include=localhost:0,1,2,3 --master_port 50123 train_re_decouple.py --cuda  --deepspeed_config config_rxrx1.json \
    --dataset rxrx1 --dataroot /raid/jiqing/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 1 \
    --env '201' --imagesize 256 --batchsize 8 \
    --nepochs 100 --nblocks 8-8-12-12-12 --factor-out --actnorm --symm-batchsize 256 \
    --task hybrid --nworkers 2 --eval-batchsize 16 --print-freq 400 --vis-freq 400 \
    --act elu --fc-end False