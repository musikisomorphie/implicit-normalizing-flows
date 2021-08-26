CUDA_VISIBLE_DEVICES=0 python train_scrc.py --cuda  --data scrc \
    --actnorm True --task classification \
    --nblocks 1-1-1 --print-freq 60 --squeeze-first True \
    --save 'experiments/scrc_class/' \
    --dataroot /home/histopath/Data/SCRC_nuclei/ --batchsize 16 --val-batchsize 32 --nepochs 100 \
    --flow reflow --classifier resnet --scale-factor 2 --env '201' --aug 'rr'  \
    --inp 'i' --oup 'cms' --couple-label False --imagesize 256 --batchsize 16 \
