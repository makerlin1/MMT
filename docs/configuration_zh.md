如何配置算子描述文件
---
## 1.修改您的各级模型
对于您自定义的模型，请改写`__repr__()`,并且要具有参数的唯一表示性，例如：
```python
    def __init__(self, ...)
     self.name = "ResNetBasicBlock-%d-%d-%d-%d-" % (in_channels, out_channels, stride, kernel)
    ...
    def __repr__(self):
        return self.name
```
其中`ResNetBasicBlock`根据不同的初始化参数(例如in_channels, out_channels等)会产生不同延迟的同类型算子，
`__repr__()`则需要唯一性的反映出对应的算子，例如：
```
ResNetBasicBlock-64-64-1-3-                                           
ResNetBasicBlock-64-64-1-7-  
...
```
这两个算子都属于ResNetBasicBlock类，但输入参数不同，算子的实际延迟是不同的。若不正确修改，则有可能在解析模型时
产生错误或预测延迟的误差较大。

## 2.配置算子描述文件
通过解析算子描述文件，`MMT`将自动生成对应算子空间的算子列表。

通常模型是由有限个算子(也包括您对nn.Module多次封装的layer)组成的，这些算子根据不同的参数配置
（例如`nn.Conv2d`的kernel_size, in_channels等参数）就可以实例化非常多的具体算子，这些具体算子在模型中的不同位置，不同输入都将会有不同的延迟。

因此，决定一个算子具体延迟的包括（算子类型，算子实例化参数，输入形状）。
需要通过以下示例中的方式表达具体的算子空间：
### 2.1 如何描述算子
* 根据以下方式描述对应算子以及可能的参数取值，`MMT`会自动遍历组合这些参数产生一系列算子，
同时也会自动过滤掉一些不合法的算子。
* 算子的第一个参数一定是决定模型输入张量形状的参数(例如nn.Conv2d的in_channels),`input_shape`必须与第一个参数的内容一一对应。
* 对于不满足上述情况的算子（例如ReLU等），可以按照示例中的方式编写对应的`input_shape`

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
* 您需要保证resnet18.py文件在运行程序时可以被引用到（您可以使用`import resnet18`测试）
* 按照下面的方式编写不同位置参数可能的取值，`MMT`会自动遍历组合这些参数产生一系列算子，
同时也会自动过滤掉一些不合法的算子。
* 算子的第一个参数一定是决定模型输入张量形状的参数(例如nn.Conv2d的in_channels),
`input_shape`必须与第一个参数的内容一一对应。

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
