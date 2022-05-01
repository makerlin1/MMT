MMT: Mnn MeTer
---
基于MNN+pytorch框架的模型延迟预测工具
## 快速开始 Quickly Start
1.编写算子描述文件 ops.yaml
```yaml
ops: None  # 没有自己定义的算子
torch.nn:
    Conv1d:
        in_channels: [8, 16, 32]  # 该参数可能的所有取值
        out_channels: [16, 32, 64]
        kernel_size: [3, 5, 7]
        stride: [1, 2]
        padding: [1, 2]
        input_shape: [[1, 8, 16], [1, 16, 16], [1, 32, 16]]  # 注意长度
    ReLU: None
    Sigmoid: None
    Linear:
        in_features: [8, 16]
        out_features: [16, 32]
        input_shape: [[1, 8], [1, 16]]
```
2.将算子导出为mnn格式
```python
generate_ops_list("ops.yaml", "/path/ops_folder")
```
### 开发日志
2022.5.1 框架设计