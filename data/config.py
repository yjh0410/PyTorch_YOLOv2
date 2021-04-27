# config.py

train_cfg = {
    'lr_epoch': (100, 150),
    'max_epoch': 200
}

# anchor size
ANCHOR_SIZE = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]

ANCHOR_SIZE_COCO = [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]

IGNORE_THRESH = 0.5
