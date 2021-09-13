deepspeed --include=localhost:0 --master_port 50123 train_re.py --cuda  --deepspeed_config config_re.json \
    --dataset scrc --dataroot /home/histopath/Data/SCRC_nuclei/ --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 0.5 \
    --env '201' --inp 'mi' --imagesize 256 --batchsize 8 \
    --nepochs 100 --nblocks 2-2-2-2 --couple-label --factor-out --actnorm \
    --task hybrid --nworkers 2 --eval-batchsize 8 --print-freq 120 --vis-freq 400
    