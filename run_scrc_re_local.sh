CUDA_VISIBLE_DEVICES=0,1 deepspeed train_img_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --actnorm True --task hybrid \
    --nblocks 1-1-1 \
    --save 'experiments/scrc_re' --imagesize 64 \
    --dataroot /home/miashan/Data/SCRC --batchsize 8 --val-batchsize 8 --nepochs 100