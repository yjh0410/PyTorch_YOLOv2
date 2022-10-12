# yolov2 config


yolov2_config = {
    'yolov2': {
        # model
        'backbone': 'darknet19',
        'pretrained': True,
        'stride': 32,  # P5
        'reorg_dim': 64,
        'head_dim': 1024,
        # anchor size
        'anchor_size': {
            'voc': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
            'coco': [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]
            },
        # matcher
        'ignore_thresh': 0.5,
        },
}