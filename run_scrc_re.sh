# CUDA_VISIBLE_DEVICES=0,1 python train_img_re.py --data scrc --actnorm True --task hybrid \
#     --nblocks 1-1-1 \
#     --save 'experiments/scrc_re' --imagesize 64 \
#     --dataroot /home/miashan/Data/SCRC_nuclei --batchsize 2 --val-batchsize 2 --nepochs 100
CUDA_VISIBLE_DEVICES=0,1 deepspeed train_img_re.py --cuda  --data scrc --deepspeed_config img_re_config.json \
    --actnorm True --task hybrid --nworkers 2 \
    --nblocks 4-4-4 \
    --save 'experiments/scrc_re' --imagesize 128 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 12 --val-batchsize 16 --nepochs 100