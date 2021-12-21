
# PyTorch_YOLOv2
这个YOLOv2项目是配合我在知乎专栏上连载的《YOLO入门教程》而创建的：

https://zhuanlan.zhihu.com/c_1364967262269693952

感兴趣的小伙伴可以配合着上面的专栏来一起学习，入门目标检测。

另外，这个项目在小batch size 的情况，如batch size=8，可能会出现nan的问题，经过其他伙伴的调试，
在batch size=8时，可以把学习率lr跳到2e-4，兴许就可以稳定炼丹啦！ 我自己训练的时候，batch size
设置为16或32，比较大，所以训练稳定。

当然，这里也诚挚推荐我的另一个YOLO项目，训练更加稳定，性能更好呦

https://github.com/yjh0410/PyTorch_YOLO-Family


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

链接：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

提取码：4la9

读者会获得 ```VOCdevkit.zip```压缩包, 分别包含 ```VOCdevkit/VOC2007``` 和 ```VOCdevkit/VOC2012```两个文件夹，分别是VOC2007数据集和VOC2012数据集.

### COCO 2017 数据集

运行 ```sh data/scripts/COCO2017.sh```，将会获得 COCO train2017, val2017, test2017三个数据集.

## 实验结果

COCO val 验证集：

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-320</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 25.8 </td><td bgcolor=white> 44.6 </td><td bgcolor=white> 25.9 </td><td bgcolor=white> 4.6 </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 47.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 29.0 </td><td bgcolor=white> 48.8 </td><td bgcolor=white> 29.7 </td><td bgcolor=white> 7.4 </td><td bgcolor=white> 31.9 </td><td bgcolor=white> 48.3 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-512</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 30.4 </td><td bgcolor=white> 51.6 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 10.1 </td><td bgcolor=white> 34.9 </td><td bgcolor=white> 46.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-544</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 30.4 </td><td bgcolor=white> 51.9 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 11.1 </td><td bgcolor=white> 35.8 </td><td bgcolor=white> 45.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 29.2 </td><td bgcolor=white> 51.6 </td><td bgcolor=white> 29.1 </td><td bgcolor=white> 13.6 </td><td bgcolor=white> 36.8 </td><td bgcolor=white> 40.5 </td></tr>
</table></tbody>


# Model

大家可以从下面的百度网盘链接来下载已训练好的模型：

链接: https://pan.baidu.com/s/1NmdqPwAmirknO5J__lg5Yw 

提起码: hlt6 
