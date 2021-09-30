deepspeed --include=localhost:0 --master_port 50123 train_re.py --cuda  --deepspeed_config config_rxrx1.json \
    --dataset rxrx1 --dataroot /home/histopath/Data/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 1 \
    --env '201' --imagesize 128 --batchsize 16 \
    --nepochs 90 --nblocks 4-8-8 --factor-out --actnorm --symm-batchsize 8 \
    --task hybrid --nworkers 2 --eval-batchsize 16 --print-freq 120 --vis-freq 20