# Pytorch-YOLOv4-ROS-lidar

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

`YOLOv4`的最小`Pytorch`实现
- YOLOv4论文 Yolo v4：https://arxiv.org/abs/2004.10934
- YOLOv4的源码地址：https://github.com/AlexeyAB/darknet
- 关于YOlOv4的更多细节： http://pjreddie.com/darknet/yolo/


- [x] 参考
- [x] 训练
    - [x] Mocaic（类似马赛克的数据增强）

```
├── README.md
├── dataset.py            # 数据集
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── demo_darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_pytorch2onnx.py  tool to convert into onnx
├── models.py             model for pytorch
├── train.py              train models.py
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch
├── data            
├── weight                --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotation.py       coco dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

![image](https://user-gold-cdn.xitu.io/2020/4/26/171b5a6c8b3bd513?w=768&h=576&f=jpeg&s=78882)

# 0. 权重下载（Weights Download）

## 0.1 darknet权重下载
- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## 0.2 pytorch权重下载
你可以使用`darknet2pytorch`把`darknet的权重`转换为`pytorch的权重`，或者直接从下面下载已经转换好的pytorch权重模型。
- baidu
    - yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) 
    - yolov4.conv.137.pth(https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA Extraction code:kcel)
- google
    - yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    - yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

# 1. 训练（Train）

[使用yolov4训练你自己的数据集](Use_yolov4_to_train_your_own_data.md)

1. 下载权重（Download weight）
2.  转换数据（Transform data）

    对于coco数据集，你可以使用`tool/coco_annotation.py`
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
3. 训练（Train）

    你可以在` cfg.py`文件中对参数进行设置
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```

# 2. 推理（Inference）

## 2.1 Performance on MS COCO dataset (using pretrained DarknetWeights from <https://github.com/AlexeyAB/darknet>)

**ONNX and TensorRT models are converted from Pytorch (TianXiaomo): Pytorch->ONNX->TensorRT.**
See following sections for more details of conversions.

- val2017 dataset (input size: 416x416)

| Model type          | AP          | AP50        | AP75        |  APS        | APM         | APL         |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DarkNet (YOLOv4 paper)|     0.471 |       0.710 |       0.510 |       0.278 |       0.525 |       0.636 |
| Pytorch (TianXiaomo)|       0.466 |       0.704 |       0.505 |       0.267 |       0.524 |       0.629 |
| TensorRT FP32 + BatchedNMSPlugin | 0.472| 0.708 |       0.511 |       0.273 |       0.530 |       0.637 |
| TensorRT FP16 + BatchedNMSPlugin | 0.472| 0.708 |       0.511 |       0.273 |       0.530 |       0.636 |

- testdev2017 dataset (input size: 416x416)

| Model type          | AP          | AP50        | AP75        |  APS        | APM         | APL         |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DarkNet (YOLOv4 paper)|     0.412 |       0.628 |       0.443 |       0.204 |       0.444 |       0.560 |
| Pytorch (TianXiaomo)|       0.404 |       0.615 |       0.436 |       0.196 |       0.438 |       0.552 |
| TensorRT FP32 + BatchedNMSPlugin | 0.412| 0.625 |       0.445 |       0.200 |       0.446 |       0.564 |
| TensorRT FP16 + BatchedNMSPlugin | 0.412| 0.625 |       0.445 |       0.200 |       0.446 |       0.563 |


## 2.2 推理图像的输入大小

- 图像的输入尺寸并不一定限制为 `320 * 320`, `416 * 416`, `512 * 512` 和`608 * 608`
- 你可以调整输入尺寸大小到不同的比例，例如：`320 * 608`
-较大的输入大小，可以帮助`检测较小的目标`，但速度就会更慢，同时GPU测内存可能会全部耗尽！

```py
height = 320 + 96 * n, n in {0, 1, 2, 3, ...}
width  = 320 + 96 * m, m in {0, 1, 2, 3, ...}
```

## 2.3 不同推理选项

- 加载`预训练的Darknet模型`和`Darknet权重`以进行推理（图像大小已在cfg文件中配置）

    ```sh
    python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile <imgFile>
    ```

- 加载`pytorch预训练模型`（`pth格式模型`）进行推理

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```
    
- 加载转换的`ONNX模型文件`进行推理（具体参考第3部分和第4部分）

- 加载转换的`TensorRT engine模型文件`进行推理（具体参考第5部分）

## 2.4 推理输出（Inference output）

这里的推理输出，主要有`两部分组成`：

- One is locations of bounding boxes, its shape is  `[batch, num_boxes, 1, 4]` which represents x1, y1, x2, y2 of each bounding box.
- The other one is scores of bounding boxes which is of shape `[batch, num_boxes, num_classes]` indicating scores of all classes for each bounding box.

Until now, still a small piece of post-processing including NMS is required. We are trying to minimize time and complexity of post-processing.


# 3. Darknet2ONNX

- **This script is to convert the official pretrained darknet model into ONNX**

- **Pytorch version Recommended:**

    - Pytorch 1.4.0 for TensorRT 7.0 and higher
    - Pytorch 1.5.0 and 1.6.0 for TensorRT 7.1.2 and higher

- **Install onnxruntime**

    ```sh
    pip install onnxruntime
    ```

- **Run python script to generate ONNX model and run the demo**

    ```sh
    python demo_darknet2onnx.py <cfgFile> <weightFile> <imageFile> <batchSize>
    ```

## 3.1 Dynamic or static batch size

- **Positive batch size will generate ONNX model of static batch size, otherwise, batch size will be dynamic**
    - Dynamic batch size will generate only one ONNX model
    - Static batch size will generate 2 ONNX models, one is for running the demo (batch_size=1)

# 4. Pytorch2ONNX

- **You can convert your trained pytorch model into ONNX using this script**

- **Pytorch version Recommended:**

    - Pytorch 1.4.0 for TensorRT 7.0 and higher
    - Pytorch 1.5.0 and 1.6.0 for TensorRT 7.1.2 and higher

- **Install onnxruntime**

    ```sh
    pip install onnxruntime
    ```

- **Run python script to generate ONNX model and run the demo**

    ```sh
    python demo_pytorch2onnx.py <weight_file> <image_path> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>
    ```

    For example:

    ```sh
    python demo_pytorch2onnx.py yolov4.pth dog.jpg 8 80 416 416
    ```

## 4.1 Dynamic or static batch size

- **Positive batch size will generate ONNX model of static batch size, otherwise, batch size will be dynamic**
    - Dynamic batch size will generate only one ONNX model
    - Static batch size will generate 2 ONNX models, one is for running the demo (batch_size=1)


# 5. ONNX2TensorRT

- **TensorRT version Recommended: 7.0, 7.1**

## 5.1 Convert from ONNX of static Batch size

- **Run the following command to convert YOLOv4 ONNX model into TensorRT engine**

    ```sh
    trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
    ```
    - Note: If you want to use int8 mode in conversion, extra int8 calibration is needed.

## 5.2 Convert from ONNX of dynamic Batch size

- **Run the following command to convert YOLOv4 ONNX model into TensorRT engine**

    ```sh
    trtexec --onnx=<onnx_file> \
    --minShapes=input:<shape_of_min_batch> --optShapes=input:<shape_of_opt_batch> --maxShapes=input:<shape_of_max_batch> \
    --workspace=<size_in_megabytes> --saveEngine=<engine_file> --fp16
    ```
- For example:

    ```sh
    trtexec --onnx=yolov4_-1_3_320_512_dynamic.onnx \
    --minShapes=input:1x3x320x512 --optShapes=input:4x3x320x512 --maxShapes=input:8x3x320x512 \
    --workspace=2048 --saveEngine=yolov4_-1_3_320_512_dynamic.engine --fp16
    ```

## 5.3 Run the demo

```sh
python demo_trt.py <tensorRT_engine_file> <input_image> <input_H> <input_W>
```

- This demo here only works when batchSize is dynamic (1 should be within dynamic range) or batchSize=1, but you can update this demo a little for other dynamic or static batch sizes.
    
- Note1: input_H and input_W should agree with the input size in the original ONNX file.
    
- Note2: extra NMS operations are needed for the tensorRT output. This demo uses python NMS code from `tool/utils.py`.


# 6. ONNX2Tensorflow

- **First:Conversion to ONNX**

    tensorflow >=2.0
    
    1: Thanks:github:https://github.com/onnx/onnx-tensorflow
    
    2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
    Run pip install -e .
    
    Note:Errors will occur when using "pip install onnx-tf", at least for me,it is recommended to use source code installation

# 7. ONNX2TensorRT and DeepStream Inference
  
  1. Compile the DeepStream Nvinfer Plugin 
  
  ```
      cd DeepStream
      make 
  ```
  2. Build a TRT Engine.
  
   For single batch, 
   ```
   trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
   ```
   
   For multi-batch, 
  ```
  trtexec --onnx=<onnx_file> --explicitBatch --shapes=input:Xx3xHxW --optShapes=input:Xx3xHxW --maxShapes=input:Xx3xHxW --minShape=input:1x3xHxW --saveEngine=<tensorRT_engine_file> --fp16
  ```
  
  Note :The maxShapes could not be larger than model original shape.
  
  3. Write the deepstream config file for the TRT Engine.
  
  
   
Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```
