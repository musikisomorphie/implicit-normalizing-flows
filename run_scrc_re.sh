CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed train_img_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --actnorm True --task density --nworkers 2 \
    --nblocks 16-16-16 --print-freq 60 --factor-out True --squeeze-first True \
    --save 'experiments/scrc_re_128' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 16 --val-batchsize 32 --nepochs 100