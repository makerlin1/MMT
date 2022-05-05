How to configure an operator description file
---
## 1.Modify your model of all levels
For your custom model, please rewrite `__repr__()` with unique representation of parameters, for example:
```python
    def __init__(self, ...)
     self.name = "ResNetBasicBlock-%d-%d-%d-%d-" % (in_channels, out_channels, stride, kernel)
    ...
    def __repr__(self):
        return self.name
```
Among them, `ResNetBasicBlock` will generate operators of the same
type with different delays according to different initialization
parameters (such as in_channels, out_channels, etc.), and `__repr__()`
needs to uniquely reflect the corresponding operators, for example:
```
ResNetBasicBlock-64-64-1-3-
ResNetBasicBlock-64-64-1-7-
...
```
These two operators belong to the ResNetBasicBlock class, but with different input parameters,
the actual delays of the operators are different. If it is not modified correctly, it is possible to
generate errors or large errors in prediction delays when analyzing the model.

## 2.Configure the operator description file
By parsing the operator description file, `MMT` will automatically
generate the operator list corresponding to the operator space.

Usually the model is composed of a limited number of operators
(including layers that you encapsulate multiple times on nn.Module),
and these operators can be instantiated according to different parameter configurations
(such as kernel_size, in_channels and other parameters of `nn.Conv2d`).
There are a lot of specific operators, and these specific operators will have
different delays for different inputs in different positions in the model.

Therefore, what determines the specific delay of an operator includes
(operator type, operator instantiation parameters, input shape).
The specific operator space needs to be expressed in the following way:
### 2.1 How to describe operators
* The corresponding operators and possible parameter value are described in the following way, then
`MMT` will automatically traverse and combine these parameters to generate a series of operators,
and will also automatically filter out some illegal operators.
* The first parameter of the operator must be a parameter that
determines the shape of the input tensor of the model (such as in_channels of nn.Conv2d),
and `input_shape` must correspond one-to-one with the content of the first parameter.
* For operators that do not meet the above conditions (such as ReLU, etc.),
the corresponding `input_shape` can be written as in the example

```yaml
torch.nn:
    Conv2d:
        in_channels: [3]
        out_channels: [64]
        kernel_size: [7]
        stride: [2]
        padding: [3]
        input_shape: [[1, 3, 224, 224]]

    BatchNorm2d:
        num_features: [64]
        input_shape: [[1, 64, 112, 112]]

    ReLU:
        no_params: true
        input_shape: [[1, 64, 112, 112]]

    AdaptiveAvgPool2d:
        output_size: [[1, 1]]
        input_shape: [[1, 512, 14, 14]]

    Linear:
        in_features: [512]
        out_features: [10]
        input_shape: [[1, 512]]
```
### 2.2 如何描述自定义的算子
例如您在resnet18.py中定义了ResNetBasicBlock与ResNetDownBlock
```python
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel=3):
    ...
class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel):
    ...
```
* You need to ensure that the resnet18.py file can be referenced when running the program (you can use `import resnet18` to test)
* Write the possible values of different positional parameters in the following way, `MMT` will automatically traverse and combine these parameters to generate a series of operators, and also automatically filter out some illegal operators.
* The first parameter of the operator must be a parameter that determines the shape of the input tensor of the model (such as in_channels of nn.Conv2d), and `input_shape` must correspond to the content of the first parameter one-to-one.

```yaml
resnet18:
    ResNetBasicBlock:
        in_channels: [64, 128, 256, 512]
        out_channels: [64, 128, 256, 512]
        stride: [1]
        kernel: [3, 5, 7]
        input_shape: [[1, 64, 112, 112], [1, 128, 56, 56], [1, 256, 28, 28], [1, 512, 14, 14]]

    ResNetDownBlock:
        in_channels: [64, 128, 256]
        out_channels: [128, 256, 512]
        stride: [[2, 1]]
        kernel: [3, 5, 7]
        input_shape: [[1, 64, 112, 112], [1, 128, 56, 56], [1, 256, 28, 28]]
```
### Some other helper skills
The `summary model` function can be used to quickly
understand the input shape of the different layers of your model
to quickly select the `input shape` of the model.
```python
...
from mmt.parser import summary_model
net = MobileNetV3(cfgs, mode='small')
ops_list = [nn.Linear, nn.Dropout, conv_3x3_bn, InvertedResidual, conv_1x1_bn, h_swish]
summary_model(net, [1, 3, 224, 224], ops_list)
print(net)
>>>
ops                                                     input_shape        out_shape
------------------------------------------------------  -----------------  -----------------
conv_3x3_bn-3-16-2                                      [1, 3, 224, 224]   [1, 16, 112, 112]
InvertedResidual-16-16-16-3-2-1-0                       [1, 16, 112, 112]  [1, 16, 56, 56]
InvertedResidual-16-72-24-3-2-0-0                       [1, 16, 56, 56]    [1, 24, 28, 28]
InvertedResidual-24-88-24-3-1-0-0                       [1, 24, 28, 28]    [1, 24, 28, 28]
InvertedResidual-24-96-40-5-2-1-1                       [1, 24, 28, 28]    [1, 40, 14, 14]
InvertedResidual-40-240-40-5-1-1-1                      [1, 40, 14, 14]    [1, 40, 14, 14]
InvertedResidual-40-240-40-5-1-1-1                      [1, 40, 14, 14]    [1, 40, 14, 14]
InvertedResidual-40-120-48-5-1-1-1                      [1, 40, 14, 14]    [1, 48, 14, 14]
InvertedResidual-48-144-48-5-1-1-1                      [1, 48, 14, 14]    [1, 48, 14, 14]
InvertedResidual-48-288-96-5-2-1-1                      [1, 48, 14, 14]    [1, 96, 7, 7]
InvertedResidual-96-576-96-5-1-1-1                      [1, 96, 7, 7]      [1, 96, 7, 7]
InvertedResidual-96-576-96-5-1-1-1                      [1, 96, 7, 7]      [1, 96, 7, 7]
conv_1x1_bn-96-576                                      [1, 96, 7, 7]      [1, 576, 7, 7]
Linear(in_features=576, out_features=1024, bias=True)   [1, 576]           [1, 1024]
h_swish                                                 [1, 1024]          [1, 1024]
Dropout(p=0.2, inplace=False)                           [1, 1024]          [1, 1024]
Linear(in_features=1024, out_features=1000, bias=True)  [1, 1024]          [1, 1000]
```



