deepspeed --include=localhost:0,1,2,3 --master_port 50123 train_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --actnorm True --task hybrid --nworkers 2 \
    --nblocks 12-12-12 --print-freq 120 --factor-out True --squeeze-first True \
    --save 'experiments/scrc_re_alpha_100/' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC/ --batchsize 16 --val-batchsize 32 --nepochs 100 \
    --env '201' --aug 'rr' --inp 'i' \
