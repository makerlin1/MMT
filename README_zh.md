![img.png](img.png)
---
用于快速构建算子延迟表并用于准确预测模型延迟的工具(基于Pytorch与MNN)
## 1.安装
MMT用于服务器端与推理端两种情况：
* 在服务器端根据指定的算子空间生成算子列表；根据算子延迟表预测给定模型的延迟。
* 在推理端根据算子列表测试算子延迟获得算子延迟表。

服务器端必须要求同时安装`Pytorch`与`MNN(C++)`,推理端必须安装`MNN(C++)`

**注意：一定要将编译`MNN`产生的`build`文件夹添加到环境变量！**

在配置好上述依赖之后，安装MMT
```
pip install mmn-meter
```
## 2.快速开始 Quickly Start
### 2.1 修改您的模型
对于您自定义的模型，请改写__repr__(),并且要具有参数的唯一表示性，例如：
```python
    def __init__(self, ...)
     self.name = "ResNetBasicBlock-%d-%d-%d-%d-" % (in_channels, out_channels, stride, kernel)
    ...
    def __repr__(self):
        return self.name
```
如果**对于不同参数输入的同类型算子，"\_\_repr\_\_()"返回的结果不能具有区分性,则非常容易造成运行错误或测量误差！**
[参考如何修改您的模型](docs/configuration_zh.md)

### 2.2 导出算子
`mmt=2.x`版本之后同时支持`描述文件生成`与`函数式生成`两种方式。
### 2.2.1 方式1：编写算子描述文件
决定一个算子具体延迟的参数包括（算子类型，算子实例化参数，输入形状）。
需要通过以下示例中的方式表达具体的算子空间：
```yaml
resnet18:
    ResNetBasicBlock:
        in_channels: [64, 128, 256, 512]
        out_channels: [64, 128, 256, 512]
        stride: [1]
        kernel: [3, 5, 7]
        input_shape: [[1, 64, 112, 112], [1, 128, 56, 56], [1, 256, 28, 28], [1, 512, 14, 14]]

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
```
[参考如何描述您的算子](docs/configuration_zh.md)

之后使用如下命令, 创建算子列表，将算子导出为mnn格式

```python
from mmt.converter import generate_ops_list

generate_ops_list("ops.yaml", "/path/ops_folder")
```
其中`ops.yaml`为算子描述文件，`/path/ops_folder`是保存算子的目录，在
该目录下会生成对应的`meta.pkl`保存有算子的元数据信息。

### 2.2.1 方式2：函数式生成
与方式一高度类似，通过使用`mmt.register`函数直接注册并生成，且支持同类型算子多次注册以减少不必要的组合
带来的冗余算子(方式一的缺点)，例如[样例](demo/MobileNetV3/convert.py)
```python
from mmt import register
import torch.nn as nn
fp = "./mbv3_ops"
reg = lambda ops, **kwargs: register(ops, fp, **kwargs)
reg(nn.Linear,
    in_features=[576, 1024],
    out_features=[1024, 1000],
    bias=[True],
    input_shape=[[1, 576], [1, 1024]],
    )
```
对比方式一：
```yaml
torch.nn:
    Linear:
        in_features: [576, 1024]
        out_features: [1024, 1000]
        bias: [True]
        input_shape: [[1, 576], [1, 1024]]
```
直接运行编写好的文件即可生成对应算子，更加详细的内容参考[样例](demo/MobileNetV3/tutorial.md)
### 2.3 在部署端记录算子的延迟，构建算子延迟表

```python
from mmt.meter import meter_ops

meter_ops("./ops", times=100)
```
`ops`为保存算子与`meta.pkl`的文件夹，`times`表示重复测试次数，运行改程序，
将统计算子的延迟，并将算子延迟表保存为`./ops/meta_latency.pkl`。该文件内具体记录了所有算子的
元数据与对应延迟。

### 2.4 在服务器端预测模型延迟

```python
from mmt.parser import predict_latency

...
model = ResNet18()
pred_latency = predict_latency(model, path, [1, 3, 224, 224], verbose=False)
```
`path`为对应`meta_latency.pkl`的路径，注意输入的张量形状必须与之前在算子描述
中设置的`input_shape`相同。

## 4.更加精准的延迟预测方法
通过测试少量模型在推理端的真实延迟与预测的延迟，`mmt`支持构建`机器学习模型`来更加准确的
预测模型延迟。具体样例参考：

## 5 如何检验MMT的预测误差
具体参考[MobileNetV3测试](demo/MobileNetV3/tutorial.md)

|Model|Num|err(%)|device|
|----|----|----|----|
|MobileNet|334|4.1%*(39%)| 40  Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz|

(*表示使用机器学习方法优化的预测器误差(只采样16个网络用于训练))
### 开发日志 & 计划
* 2022.5.1 框架设计
* 2022.5.2 完成算子生成与测算
* 2022.5.3 完成模型预测与检验, 完善代码介绍
* 2022.5.4 上线pypi