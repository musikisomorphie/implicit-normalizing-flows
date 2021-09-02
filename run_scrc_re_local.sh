deepspeed --include=localhost:0 --master_port 50123 train_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --save 'experiments/scrc_resize64_circle/' \
    --dataroot /home/histopath/Data/SCRC_nuclei/ \
    --flow reflow --classifier resnet --scale-factor 2 --env '201' --aug 'r'  \
    --inp 'i' --oup 'cms' --couple-label False --imagesize 64 --batchsize 8 \
    --actnorm True --task hybrid --nworkers 2 --val-batchsize 8 --nepochs 100 \
    --nblocks 16-16-16 --print-freq 120 --factor-out True --squeeze-first True