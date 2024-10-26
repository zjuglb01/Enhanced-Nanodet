# Enhanced Nanodet

## 1.Introduction

Enhanced Nanodet is an improved version of nanodet-plus, which has 0.77M parameters and 28.1 coco-map with 320*320 resolution. Enhanced NanoDet is lighter than previous NanoDet but its performance is rather OK. 

The most codes are from the NanoDet-plus(https://github.com/RangiLyu/nanodet). Very thanks the authors and codes of NanoDet-plus. 

## 2. Detail

### 2.1 neck improvement

We proposed a ghost2former block, which is used to replace the ghost_res of NanoDet-plus.

### 2.2 backbone improvement
MobileNeXt is used to replace the ShuffleNetV2.

### 2.3 3 * 3depth-wise convolution improvement
A kernel refine(KR) menthod is proposed to improve the feature extracting performance of 3 * 3 depth-wise convolution.

## 3. Usage

### 3.1 config
if you want the ghost2former block, you need set the 'ghost_block_type' to be 'plus' in the config file.

if you want the ghost_res block of nanodet-plus, you need set the 'ghost_block_type' to be 'normal' in the config file.

if you want the MobileNeXt as the backbone, you need set the backbone's name to be 'mobilenext' in the config file.

The KR module is enabled when you use MobileNeXt as backbone.

### 3.2 dataset

Use coco dataset for example. Download coco dataset from http://images.cocodataset.org/ .

Then set the image path and label path in the config file. 

If you want to train and test VOC dataset, you can refer this config file: ./config/nanodet_custom_xml_dataset.yml



### 3.3 training

```shell
python ./tools/train.py ./config/xxx.yml

```

### 3.4 testing
```shell
 python ./tools/test.py --config ./config/xxx.yml  --model checkpoint_path

```
### 3.5 onnx file export
```shell
python ./tools/export_onnx.py --cfg_path ./config/xxxxxx.yml \
                              --out_path ./xxxxx_model.onnx  \
                              --model_path ./xxxxx/model_best.ckpt \
                              --input_shape 320,320
```
### 3.6 onnx inference
```shell
 python ./tools/onnx_infer.py   --model_path  ./xxxxx/model.onnx \
                                --config  ./xxx.yml
```

### 3.7 FLOPs and parameters 
```shell
python ./tools/flops.py ./xxx.tml  --input_shape 320,320
```

### 3.8 draw heatmap
```shell
python ./tools/grad_cam.py --config ./config/xxx.yml \
                           --model ./workspace/xxxxx.pth \
                           --img ./xxxxx.jpg
```
## 4. performance

We provide the onnx checkpoint in weight folder  

Model           |     Backbone      |  neck        |Resolution | COCO mAP | FLOPS | Params(M) |cpu-time|
:--------------:|:-----------------:|:------------:|:----------:|:--------:|:-----:|:------:|:--:|
NanoDet         | ShuffleNetV2 1.0x |  ghost_res   |320*320   |   20.6   | 0.72G | 0.95   |  -  |
NanoDet-Plus    | ShuffleNetV2 1.0x |  ghost_res   |320*320   |   27.0   | 0.90G | 1.17   | 6ms |
NanoDet-Plus    | ShuffleNetV2 1.0x |  ghost_res   |416*416   |   30.4   | 1.52G | 1.17   | 9ms |
NanoDet-Plus    | ShuffleNetV2 1.5x |  ghost_res   |416*416   |   34.1   | 2.97G | 2.44   | 12ms|
Enhanced NanoDet_s     |  MobileNeXt       | ghost2former |320*320   |   28.1   | 0.63G | 0.77   | 6ms |
Enhanced NanoDet_m     |  MobileNeXt       | ghost2former |416*416   |   33.9   | 1.30G | 1.25   | 9ms|
Enhanced NanoDet_l     |  MobileNeXt       | ghost2former |416*416   |   36.5   | 2.03G | 2.54   | 13ms|
Enhanced NanoDet_l     |  MobileNeXt       | ghost2former |640*640   |   40.5   | - | 2.54   | - |



Compare with other 1M detectors. We achieve the highest coco-map with only 320*320 resolution.

Model           | Resolution | COCO mAP | FLOPS | Params(M) |link|
:--------------:|:----------:|:--------:|:-----:|:---------:|:--:|
NanoDet-m       |  320*320   |   20.6   | 0.72G | 0.95   |https://github.com/RangiLyu/nanodet |
Enhanced NanoDet_s       |  320*320   |   28.1   | 0.63G | 0.77   | -  |
FastestDet      |  352*352   |   13.0   |  -    | 0.24   |https://github.com/dog-qiuqiu/FastestDet |
PP-PicoDet-S    |  320*320   |   27.1   | 0.73G | 0.99   |https://github.com/PaddlePaddle/PaddleDetection |
PicoDet-XS      |  320*320   |   23.5   | 0.67G | 0.70   |https://github.com/CycloneBoy/PPDetectionPytorch/tree/master/configs/picodet |
FemtoDet        |  640*640   |   6.2    |  -    | 0.07   |https://github.com/yh-pengtu/FemtoDet     |
YOLOX-Nano      |  416*416   |   25.8   | 1.08G | 0.91   |https://github.com/Megvii-BaseDetection/YOLOX |
Tiny-DSOD       |  300*300   |   23.2   | 1.12G | 0.95   |https://github.com/lyxok1/Tiny-DSOD  |

Compare with detectors more than 1M parameters.
Model           | Resolution | COCO mAP | FLOPS | Params(M) |link|
:--------------:|:----------:|:--------:|:-----:|:---------:|:--:|
Enhanced NanoDet_l|416*416   |   36.5   | 2.03G | 2.54   | -|
NanoDet-plus1.0x|  416*416   |   30.4   | 1.52G | 1.17   |https://github.com/RangiLyu/nanodet |
NanoDet-plus1.5x|  416*416   |   34.1   | 2.97G | 2.44   |https://github.com/RangiLyu/nanodet |
yolov6n         |  416*416   |   30.8   |  -    | 4.30   |https://github.com/meituan/YOLOv6 |
yolov8n         |  416*416   |   32.7   |   -   | 3.20   | -|
yolov7t         |  416*416   |   35.2   |   -   | 6.20   | -|
PP-PicoDet-M    |  416*416   |   34.3   | 2.50G | 2.15   |https://github.com/PaddlePaddle/PaddleDetection |
PP-PicoDet-L    |  640*640   |   40.9   | 8.91G | 3.30   |https://github.com/PaddlePaddle/PaddleDetection |
PicoDet-S       |  416*416   |   32.5   | 1.65G | 1.18   |https://github.com/CycloneBoy/PPDetectionPytorch/tree/master/configs/picodet |
PicoDet-M       |  416*416   |   37.5   | 4.34G | 3.46   |https://github.com/CycloneBoy/PPDetectionPytorch/tree/master/configs/picodet |
PicoDet-L       |  416*416   |   39.4   | 7.10G | 5.80   |https://github.com/CycloneBoy/PPDetectionPytorch/tree/master/configs/picodet |
YOLOX-t         |  416*416   |   32.8   | 6.45G | 5.06   |https://github.com/Megvii-BaseDetection/YOLOX |
YOLOX-s         |  416*416   |   36.5   |   -   | 9.00   |https://github.com/Megvii-BaseDetection/YOLOX |
DPNet           |  320*320   |   30.5   | 1.04G | 2.50   |https://arxiv.org/pdf/2209.13933v1.pdf|
MobileDets      |  320*320   |   26.9   | 1.43G | 4.85   |https://github.com/tensorflow/models|
ThunderNet      |  320*320   |   28.1   | 1.30G | -      |http://openaccess.thecvf.com/content_ICCV_2019/papers/Qin_ThunderNet_Towards_Real-Time_Generic_Object_Detection_on_Mobile_Devices_ICCV_2019_paper.pdf
Pelee           |  304*304   |    22.4   |1.29G | 6.0    |https://arxiv.org/pdf/1804.06882v3.pdf|


PS:
1. The cpu time is tested on i9-10900k with onnxruntime tools.
2. In the future, we will provide bigger size model with larger resolution. 
3. The details of experiments can be referred from this link:https://p6bivngg5c.feishu.cn/docx/Qovmdfd9hoVOsFxdZ7HcVkBRnjc

## 5. Acknowledge

https://github.com/RangiLyu/nanodet

https://github.com/yitu-opensource/MobileNeXt

## 6. Cited
If you use Enhanced Nanodet in your research, please cite our work and give a star ⭐:
```
 @misc{zjuglb01Enhanced-Nanodet
  title = {Enhanced NanoDet: A lighter but stronger nano detector},
  author = {zjuglb01},
  year={2024}
}
```