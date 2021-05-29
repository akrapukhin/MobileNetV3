# MobileNetV3
An implementation of the MobileNetV3 models in Pytorch with scripts for training, testing and measuring latency.

## MobileNets Summary
MobileNetV1
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
[https://arxiv.org/pdf/1704.04861.pdf]
The neural net uses deapthwise separable convolutions instead of standard convolutions. In standard convolutions each filter has the same number of channels as the input tensor it is applied to. In deapthwise separable convolutions each channel of the input tensor is filtered by a 1-channel filter (depthwise convolution), and then the resulting features are combined using multichannel 1x1 filters (pointwise convolution). It allows to significantly reduce the number of operations and parameters as well as lower the latency. Accuracy almost stays at the same level. Two hyperparameters are used to trade off accuracy for size and latency: width multiplier (uniformly changes the number of channels inside the layers) and resolution multiplier (changes the resolution of input images).

MobileNetV2
MobileNetV2: Inverted Residuals and Linear Bottlenecks
[https://arxiv.org/pdf/1801.04381.pdf]
Based on the inverted residual with linear bottleneck module. This module takes as an input a compressed tensor with small number of channels, expands it using 1x1 convolutions (increasing the number of channels), filters it using 1-channel filters (depthwise convolution as in V1) and then compresses it using 1x1 convolution. Non-linearity is not applied to the output of the last 1x1 compressing convolution (hence the name "linear bottleneck"). The output of the module is then feeded to the next module. The shortcuts are applied to the bottleneck. Using inverted residuals and linear bottlenecks allows to decrease the number of parameters and operations, and reduce latency compared to MobileNetV1. Accuracy stays at the same level.

MobileNetV3
Searching for MobileNetV3
[https://arxiv.org/pdf/1905.02244.pdf]
Authors present two models which are targeted for high and low resource use cases - MobileNetV3-Large and MobileNetV3-Small. The models are created using automated architecture search algorithms and then improved through novel architecture techniques. First, platform-aware neural architecture search (NAS) is used to optimize network blocks. Second, NetAdapt is utilized to search for optimal number of filters in each layer. Then, in the resulting models authors halve the number of filters in the first layer and move the final convolution layer past the average pooling layer to reduce the latency. Also, hard swish is used instead of Relu in the second half of the net to improve accuracy. Moreover, the squeeze-and-excite block is used, which is basically an attention mechanism allowing the net to learn to amplify important channels and attenuate less important ones. As a result, the model consists of practically the same blocks as V2 - 1x1 expansion convolution, depthwise convolution, 1x1 compression convolution, shortcut. The differences are in using hard swish instead of Relu and in applying squeeze-and-excite to the output of the depthwise convolution layer. V3 is faster and more accurate than V2.

## Requirements
You need to install Pytorch with torchvision. https://pytorch.org/

## Training
The nets are trained on the CIFAR100 dataset. 
To start training, use the following command:
python train.py
By default, the small model is trained with width multiplier 1.0 for 20 iterations with batch size 128. 
You can set these parameters using the following flags:

--model
--width
--iter
--batch

Use python train.py -h to see help info

If your Pytorch installation comes with CUDA support and CUDA is avaliable on your machine the training will be done on a GPU. Otherwise, CPU will be used. 

I got the following results after training for 20 iterations:
    Mobilenet_v3_small_0.25 val_acc: 60.13%
    Mobilenet_v3_small_0.50 val_acc: 66.37%
    Mobilenet_v3_small_1.00 val_acc: 69.44%
    Mobilenet_v3_large_0.25 val_acc: 65.03%
    Mobilenet_v3_large_0.50 val_acc: 69.50%
    Mobilenet_v3_large_1.00 val_acc: 71.75%


## Testing
python test.py
For each model present in the trained_models folder the script will do classification of all images from the test_images folder. There is already one trained model in trained_models and 20 images in test_images.


## Latency measurements
python latency.py
The script will measure the latency of the small and large models with different width multipliers (0.25, 0.5, 1.0) on both CPU and GPU (if available).

## Notes
- Stride in the initial layers is set to 1 by default instead of 2 to adapt for small 32x32 resolution of CIFAR. If you want to use stride 2, set the si parameter of the network to 2 at initialization. Example:
Mobilenet_v3_large(wm=1.0, si=2)
- 