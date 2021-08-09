CUDA_VISIBLE_DEVICES=4 python train_scrc.py --data scrc --actnorm True --task hybrid \
    --nblocks 1-1-1 --print-freq 60 \
    --save 'experiments/scrc_tp_rc_rr_he_correct' --imagesize 256 \
    --dataroot /raid/jiqing/Data/SCRC --batchsize 32 --val-batchsize 64 --nepochs 100
