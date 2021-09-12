deepspeed --include=localhost:0 --master_port 50123 train_re.py --cuda  --data scrc --deepspeed_config config_re.json \
    --save 'experiments/scrc_resize128_circle_add_res/' \
    --dataroot /home/histopath/Data/SCRC_nuclei/ \
    --flow reflow --classifier resnet --shuffle-factor 2 --env '201' --aug 'rr'  \
    --inp 'im' --oup 'cms' --couple-label False --imagesize 128 --batchsize 8 \
    --actnorm True --task hybrid --nworkers 2 --eval-batchsize 8 --nepochs 200 \
    --nblocks 4-4-4 --print-freq 120 --factor-out True --squeeze-first True \
    --right-pad 0