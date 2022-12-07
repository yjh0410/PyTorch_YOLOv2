python train.py \
        --cuda \
        -d coco \
        -ms \
        -bs 16 \
        -accu 4 \
        --lr 0.001 \
        --max_epoch 200 \
        --lr_epoch 100 150 \
        