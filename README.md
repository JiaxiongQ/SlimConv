# SlimConv
This repository contains the code (in PyTorch) for "[SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping](https://arxiv.org/pdf/2003.07469.pdf)" paper (ArXiv) or "SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Features Reconstructing" paper

## Requirements
- [Python3.6](https://www.python.org/downloads/)
- [PyTorch(1.1.0+)](http://pytorch.org)

## Pretrained models on ImageNet
Some of pretrained models are released in [Google Drive](https://drive.google.com/drive/folders/1Dalp1v18FtjcZVXsBpyr3gZ2CT8_Ba62?usp=sharing), which including Sc-ResNet-50, Sc-ResNet-50(cosine), Sc-ResNet-101, Sc-ResNet-50(k=8/3) and Sc-ResNeXt-101(32x3d).

## Note
You can use our module on your own tasks to reduce parameters, FLOPs and improve the performance. 

Just replace 3x3_conv with slim_conv_3x3 and change the input channel number of the next conv layer.

## Comparison with SOTA on ImageNet
Y:Yes, N:No. We use the tool supplied by [DMCP](https://github.com/Zx55/dmcp) to count FLOPs here.

|    Method                        | Manual |    Top-1 Error    |    FLOPs(10^9)    |    Params(10^6)    |
|----------------------------------|--------|-------------------|-------------------|--------------------|
| Sc-ResNeXt-101(32x3d, k=2)(ours) |  Y     |    21.18          |    4.58           |    23.70           |
|                                  |        |                   |                   |                    |
| [DMCP-ResNet-50](http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_DMCP_Differentiable_Markov_Channel_Pruning_for_Neural_Networks_CVPR_2020_paper.pdf)                   |  N     |    23.50          |    2.80           |    23.18           |
| Sc-ResNet-50(k=4/3)(ours)        |  Y     |    22.77          |    2.65           |    16.76           |
|                                  |        |                   |                   |                    |
| [DMCP-ResNet-50](http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_DMCP_Differentiable_Markov_Channel_Pruning_for_Neural_Networks_CVPR_2020_paper.pdf)                   |  N     |    25.90          |    1.10           |    14.47           |
| [Ghost-ResNet-50 (s=2)](http://openaccess.thecvf.com/content_CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf)            |  Y     |    24.99          |    2.15           |    13.95           |
| Sc-ResNet-50(k=8/3)(ours)        |  Y     |    24.48          |    1.88           |    12.10           |

## Compressed ratio of ResNet-50 with SlimConv on CIFAR-100
Just adjust k of SimConv. 
![image](https://github.com/JiaxiongQ/SlimConv/blob/master/compress.png)                                                                                                                                         
## Citation 
If you use our code or method in your work, please cite the following:
```
@article{Qiu2020SlimConv,
  title={SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping},
  author={Jiaxiong Qiu and Cai Chen and Shuaicheng Liu and Bing Zeng},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.07469}
}
```
Please direct any questions to [Jiaxiong Qiu](https://jiaxiongq.github.io/) at qiujiaxiong727@gmail.com



