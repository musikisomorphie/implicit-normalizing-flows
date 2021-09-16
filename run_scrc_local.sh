deepspeed --include=localhost:0 --master_port 50123 train_re_decompose.py --cuda  --deepspeed_config config_re.json \
    --dataset scrc --dataroot /home/histopath/Data/SCRC_nuclei --save 'experiments/' \
    --flow reflow --classifier resnet --shuffle-factor 2 --scale-factor 0.25 \
    --env '201' --imagesize 256 --batchsize 16 \
    --nepochs 100 --nblocks 8-8-8 --factor-out --actnorm \
    --task hybrid --nworkers 2 --eval-batchsize 16 --print-freq 120 --vis-freq 400
    