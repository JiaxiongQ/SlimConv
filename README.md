# SlimConv
This repository contains the code (in PyTorch) for "[SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping](https://arxiv.org/pdf/2003.07469.pdf)" paper (ArXiv)

## Note
The pretrained models on ImageNet will be released soon, you can use our module on your own tasks to reduce parameters, FLOPs and improve the performance. 

Just replace 3x3_conv with slim_conv_3x3 and change the input channel number of the next conv layer.

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



