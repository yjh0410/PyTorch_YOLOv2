
# PyTorch_YOLOv2
这个YOLOv2项目是配合我在知乎专栏上连载的《YOLO入门教程》而创建的：

https://zhuanlan.zhihu.com/c_1364967262269693952

感兴趣的小伙伴可以配合着上面的专栏来一起学习，入门目标检测。

另外，这个项目在小batch size 的情况，如batch size=8，可能会出现nan的问题，经过其他伙伴的调试，
在batch size=8时，可以把学习率lr跳到2e-4，兴许就可以稳定炼丹啦！ 我自己训练的时候，batch size
设置为16或32，比较大，所以训练稳定。

当然，这里也诚挚推荐我的另一个YOLO项目，训练更加稳定，性能更好呦

https://github.com/yjh0410/PyTorch_YOLO-Family

# 配置环境
- 我们建议使用anaconda来创建虚拟环境:
```Shell
conda create -n yolo python=3.6
```

- 然后，激活虚拟环境:
```Shell
conda activate yolo
```

- 配置环境:
运行下方的命令即可一键配置相关的深度学习环境：
```Shell
pip install -r requirements.txt 
```
如果您已经学习了笔者之前的YOLOv1项目，那么就不需要再次创建该虚拟环境了，二者的环境是可以共用的。

## 训练所使用的tricks

- [x] batch norm
- [x] hi-res classifier
- [x] convolutional
- [x] anchor boxes
- [x] better backbone: resnet50
- [x] dimension priors
- [x] location prediction
- [x] passthrough
- [x] multi-scale
- [x] hi-red detector

## 数据集

### VOC2007与VOC2012数据集

读者可以从下面的百度网盘链接来下载VOC2007和VOC2012数据集

链接：https://pan.baidu.com/s/1qClcQXSXjP8FEnsP_RrZjg 

提取码：zrcj 

读者会获得 ```VOCdevkit.zip```压缩包, 分别包含 ```VOCdevkit/VOC2007``` 和 ```VOCdevkit/VOC2012```两个文件夹，分别是VOC2007数据集和VOC2012数据集.

### COCO 2017 数据集

* 自己下载

运行 ```sh data/scripts/COCO2017.sh```，将会获得 COCO train2017, val2017, test2017三个数据集.

* 百度网盘下载：

这里，笔者也提供了由笔者下好的COCO数据集的百度网盘链接：

链接：https://pan.baidu.com/s/1XQqeHgNMp8U-ohbEWuT2CA 

提取码：l1e5

## 实验结果
VOC2007 test 测试集

| Model             |  Input size  |   mAP   | Weight |
|-------------------|--------------|---------|--------|
| YOLOv2            |  320×320     |   73.4  |    -   |
| YOLOv2            |  416×416     |   77.1  |    -   |
| YOLOv2            |  512×512     |   78.0  |    -   |
| YOLOv2            |  608×608     |   78.3  | [github](https://github.com/yjh0410/PyTorch_YOLOv2/releases/download/yolov2_weight/yolov2_voc.pth) |


COCO val 验证集

| Model             |  Input size    |   AP    |   AP50    | Weight|
|-------------------|----------------|---------|-----------|-------|
| YOLOv2            |  320×320       |  24.1   |   42.8    |   -   |
| YOLOv2            |  416×416       |  27.2   |   47.3    |   -   |
| YOLOv2            |  512×512       |  28.8   |   50.0    |   -   |
| YOLOv2            |  608×608       |  29.7   |   51.7    | [github](https://github.com/yjh0410/PyTorch_YOLOv2/releases/download/yolov2_weight/yolov2_coco.pth) |

大家可以点击表格中的[github]()来下载模型权重文件。

# 训练模型
运行下方的命令可开始在```VOC```数据集上进行训练：
```Shell
python train.py \
        --cuda \
        -d voc \
        -ms \
        --batch_size 32 \
        --lr 0.001 \
        --max_epoch 200 \
        --lr_epoch 100 150 \
```

# 测试模型
运行下方的命令可开始在```VOC```数据集上进行训练：
```Shell
python test.py \
        --cuda \
        -d voc \
        -size 416 \
        --weight path/to/weight \
```


# 验证模型
运行下方的命令可开始在```VOC```数据集上进行训练：
```Shell
python eval.py \
        --cuda \
        -d voc \
        -size 416 \
        --weight path/to/weight \
```