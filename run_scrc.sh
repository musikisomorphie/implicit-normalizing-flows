CUDA_VISIBLE_DEVICES=0 python train_scrc.py --cuda  --data scrc \
    --actnorm True --task classification \
    --nblocks 1-1-1 --print-freq 60 --squeeze-first True \
    --save 'experiments/scrc_256_64/' --imagesize 256 \
    --dataroot /home/histopath/Data/SCRC_nuclei/ --batchsize 32 --val-batchsize 64 --nepochs 100 \
    --env '201' --aug 'rr' --inp 'i'
