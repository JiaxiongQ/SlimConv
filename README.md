# SlimConv
This repository contains the code (in PyTorch) for "[SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping](https://arxiv.org/pdf/2003.07469.pdf)" paper (ArXiv)

## Note
The pretrained models on ImageNet will be released soon, you can use our module on your own tasks to reduce parameters, FLOPs and improve the performance. 

Just replace 3x3_conv with slim_conv_3x3 and change the input channel number of the next conv layer.

## Comparison with SOTA on ImageNet
Y:Yes N:No

|    Method                        | Manual |    Top-1 Error    |    FLOPs(10^9)    |    Params(10^6)    |
|----------------------------------|--------|-------------------|-------------------|--------------------|
| Sc-ResNeXt-101(32x3d, k=2)(ours) |  Y     |    21.18          |    4.64           |    23.70           |
|                                  |        |                   |                   |                    |
| DMCP-ResNet-50                   |  N     |    23.50          |    2.80           |    23.18           |
| Sc-ResNet-50(k=4/3)(ours)        |  Y     |    23.29          |    2.69           |    16.76           |
|                                  |        |                   |                   |                    |
| DMCP-ResNet-50                   |  N     |    25.90          |    1.10           |    14.47           |
| Ghost-ResNet-50 (s=2)            |  Y     |    24.99          |    2.18           |    13.95           |
| Sc-ResNet-50(k=8/3)(ours)        |  Y     |    24.48          |    1.91           |    12.10           |

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



