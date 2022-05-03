![img.png](img.png)
## Intro
基于MNN+pytorch框架的模型延迟预测工具
## 1.安装
先完成MNN编译，以及pytorch等依赖
### 1.1 MNN 安装
1.cmake
```
pip install cmake
```
2.protobuf [参考指南](https://zhuanlan.zhihu.com/p/160249058)

其他具体参考[官方文档](https://www.yuque.com/mnn/cn/model_convert)

**注意：一定要将build文件夹添加到环境变量！**

## 2.快速开始 Quickly Start
1.编写算子描述文件
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

    MaxPool2d:
        kernel_size: [3]
        stride: [2]
        padding: [1]
        input_shape: [[1, 64, 112, 112]]

    AdaptiveAvgPool2d:
        output_size: [[1, 1]]
        input_shape: [[1, 512, 14, 14]]

    Linear:
        in_features: [512]
        out_features: [10]
        input_shape: [[1, 512]]
```
2.将算子导出为mnn格式
```python
from core.converter import generate_ops_list
generate_ops_list("ops.yaml", "/path/ops_folder")
```
将生成对应算子至文件夹内

3.在目标硬件上测试性能
```python
from core.meter import meter_ops
meter_ops("./ops", times=100)
```

### 开发日志
* 2022.5.1 框架设计
* 2022.5.2 完成算子生成与测算
* 2022.5.3 完成模型预测与检验