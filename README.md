# Delving into Shape-aware Zero-shot Semantic Segmentation


Paper Accepted to CVPR 2023.

![Alt text](./main.png)


Thanks to the impressive progress of large-scale vision-language pretraining, recent recognition models can classify arbitrary objects in a zero-shot and open-set manner,with a surprisingly high accuracy. However, translating this success to semantic segmentation is not trivial, because this dense prediction task requires not only accurate semantic understanding but also fine shape delineation and existing vision-language models are trained with image-level language descriptions. To bridge this gap, we pursue shape-aware zero-shot semantic segmentation in this study. Inspired by classical spectral methods in the image segmentation literature, we propose to leverage the eigenvectors of Laplacian matrices constructed with self-supervised pixel-wise features to promote shape-awareness. Despite that this simple and effective technique does not make use of the masks of seen classes at all, we demonstrate that it outperforms a state-of-the-art shape-aware formulation that aligns ground truth and predicted edges during training. We also delve into the performance gains achieved on different datasets using different backbones and draw several interesting and conclusive observations: the benefits of promoting shape-awareness highly relates to mask compactness and language embedding locality. Finally, our method sets new state-of-the-art performance for zero-shot semantic segmentation on both Pascal and COCO, with significant margins. 
   
## Installation

### Requirements
torch==1.7.1
torchvision==0.8.2 
timm==0.4.12
opencv-python==4.1.1


### Data preparation
Download the PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> datasets following  [HERE](https://github.com/juhongm999/hsnet).  


The ./datasets/ folder should have the following hierarchy:


    └── datasets/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   ├── SegmentationClassAug/
        │   └── pascal_k5/
        ├── COCO2014/           
        │   ├── annotations/
        │   │   ├── train2014/  # (dir.) training masks
        │   │   ├── val2014/    # (dir.) validation masks 
        │   │   └── ..some json files..
        │   ├── train2014/
        │   ├── val2014/
        │   └── coco_k5/
        

#### Eigenvector 
Download the top K=5 eigenvectors of the Laplacian matrix of image features from [HERE](labelmaterial.s3.amazonaws.com/release/iiw_dataset-release-0.zip). Unzip it directly and merge them with the current ./datasets/pascal_k5/  and  ./datasets/coco_k5/ folder

## Training and evaluating

#### Training 

>CUDA_VISIBLE_DEVICES=0,1,2,3  python sazs.py train
>                --arch {drn_d_105, vitl16_384}  
>                --fold {0, 1, 2, 3}  --batch_size 6
>                --random-scale 2 --random-rotate 10 
>                --lr 0.0002 --drate 0.9 --lr-mode poly
>                --benchmark {pascal, coco}



#### Evaluating

 To test the trained model with its checkpoint:


>CUDA_VISIBLE_DEVICES=0  python sazs.py test
>                --arch {drn_d_105, vitl16_384}  
>                --fold {0, 1, 2, 3}  --batch_size 1 
>                --benchmark {pascal, coco}
>                --eig_dir ./datasets/{pascal, coco}_k5/
>                --resume "path_to_trained_model/best_model.pt"






## Pretrained model
You can download the pre-trained shapenet model from  [HERE](https://pan.baidu.com/s/15OgcaOSwDpAEO6XRu-IdyA?pwd=x9w0) (extract code: x9w0) and our corresponding pre-trained models are available as follows:. 

##### PASCAL-5<sup>i</sup>
<table>
  <thead>
    <tr style="text-align: right;">
       <th>Dataset</th>
      <th>Fold</th>
      <th>Backbone</th>
      <th>Text Encoder</th>
      <th>mIoU</th>
      <th>URL</th>
      <th>Extract Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
       <th>PASCAL</th>
       <td>0</td>
      <th>ViT-L/16</th>
      <th>ViT-B/32</th>
      <th>62.7</th>
      <td><a href="https://pan.baidu.com/s/1fmH6g-RgKtWWuRNJSqzJjg?pwd=uvki">download</a></td>
      <th>uvki</th>
    </tr>
    <tr>
       <th>PASCAL</th>
       <td>1</td>
      <th>ViT-L/16</th>
      <th> ViT-B/32</th>
      <th>64.3</th>
      <td><a href="https://pan.baidu.com/s/1XmT31qoCGkQo3m_-pM9x2w?pwd=udnq">download</a></td>
      <th>udnq</th>
    </tr>
    <tr>
       <th>PASCAL</th>
       <td>2</td>
      <th>ViT-L/16</th>
      <th>ViT-B/32</th>
      <th>60.6</th>
      <td><a href="https://pan.baidu.com/s/1GD3wS3OhfLcW4xNT3OIA1Q?pwd=yh37">download</a></td>
      <th>yh37</th>
    </tr>
    <tr>
       <th>PASCAL</th>
       <td>3</td>
      <th>ViT-L/16</th>
      <th> ViT-B/32</th>
      <th>50.2</th>
      <td><a href="https://pan.baidu.com/s/15u8bC5y8rx2IHGT0zkPlHg?pwd=8wp3">download</a></td>
      <th>8wp3</th>
  </tbody>
</table>

##### COCO-20<sup>i</sup> 
<table>
  <thead>
    <tr style="text-align: right;">
       <th>Dataset</th>
      <th>Fold</th>
      <th>Backbone</th>
      <th>Text Encoder</th>
      <th>mIoU</th>
      <th>URL</th>
      <th>Extract Code</th>
    </tr>
  </thead>
  <tbody>
    </tr>
    <tr>
       <th>COCO</th>
       <td>0</td>
      <th>ViT-L/16</th>
      <th>ViT-B/32</th>
      <th>33.8</th>
      <td><a href="https://pan.baidu.com/s/1MUeYzHsY7l5jeXNA2HWlQw?pwd=z531">download</a></td>
      <th>z531</th>
    </tr>
    <tr>
       <th>COCO</th>
       <td>1</td>
      <th>ViT-L/16</th>
      <th>ViT-B/32</th>
      <th>38.1</th>
      <td><a href="https://pan.baidu.com/s/1CEGnwy79dT5AxVpfdt2n2g?pwd=hjcw">download</a></td>
       <th>hjcw</th>
    </tr>
    <tr>
       <th>COCO</th>
       <td>2</td>
      <th>ViT-L/16</th>
      <th>ViT-B/32</th>
      <th>34.4</th>
      <td><a href="https://pan.baidu.com/s/10TSDLmy2N-Qrhl9GMS7M7w?pwd=kghz">download</a></td>
      <th>kghz</th>
    </tr>
    <tr>
       <th>COCO</th>
       <td>3</td>
      <th>ViT-L/16</th>
      <th>ViT-B/32</th>
      <th>35.0</th>
      <td><a href="https://pan.baidu.com/s/1AVjTMW4aM1s0qBblH16RDA?pwd=60uo">download</a></td>
       <th>60uo</th>
      </tbody>
</table>

<table>
  <thead>
    <tr style="text-align: right;">
       <th>Dataset</th>
      <th>Fold</th>
      <th>Backbone</th>
      <th>Text Encoder</th>
      <th>mIoU</th>
      <th>URL</th>
      <th>Extract Code</th>
    </tr>
  </thead>
  <tbody>
    </tr>
    <tr>
       <th>COCO</th>
       <td>0</td>
      <th>DRN</th>
      <th>ViT-B/32</th>
      <th>33.8</th>
      <td><a href="https://pan.baidu.com/s/1MUeYzHsY7l5jeXNA2HWlQw?pwd=z531">download</a></td>
      <th>z531</th>
    </tr>
    <tr>
       <th>COCO</th>
       <td>1</td>
      <th>DRN</th>
      <th>ViT-B/32</th>
      <th>38.1</th>
      <td><a href="https://pan.baidu.com/s/1CEGnwy79dT5AxVpfdt2n2g?pwd=hjcw">download</a></td>
       <th>hjcw</th>
    </tr>
    <tr>
       <th>COCO</th>
       <td>2</td>
      <th>DRN</th>
      <th>ViT-B/32</th>
      <th>34.4</th>
      <td><a href="https://pan.baidu.com/s/10TSDLmy2N-Qrhl9GMS7M7w?pwd=kghz">download</a></td>
      <th>kghz</th>
    </tr>
    <tr>
       <th>COCO</th>
       <td>3</td>
      <th>DRN</th>
      <th>ViT-B/32</th>
      <th>35.0</th>
      <td><a href="https://pan.baidu.com/s/1AVjTMW4aM1s0qBblH16RDA?pwd=60uo">download</a></td>
       <th>60uo</th>
      </tbody>
</table>
