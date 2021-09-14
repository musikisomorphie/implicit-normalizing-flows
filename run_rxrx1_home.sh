deepspeed --include=localhost:0,1 --master_port 50123 train_re.py --cuda  --deepspeed_config config_re.json \
    --dataset rxrx1 --dataroot /home/miashan/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 0.5 \
    --env '201' --imagesize 256 --batchsize 8 \
    --nepochs 100 --nblocks 2-2-2-2 --factor-out --actnorm \
    --task hybrid --nworkers 2 --eval-batchsize 8 --print-freq 120 --vis-freq 400
    