Tutorial for MobileNetV3 Benchmark
---
## Step 1, prepare your models
You should modify the `__repr__()`method of your defined models
like:
```python
    self.repr = "InvertedResidual-%d-%d-%d-%d-%d-%d-%d" % (
            inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs)
...
def __repr__(self):
    return self.repr
```
You also need to modify the final model in order to test the different configs.
```python
class MobileNetV3(nn.Module):
    ...
    def __repr__(self):
        return "MobileNetV3" + str(self.id)
```
Here we set a new attribute (self.id) to represent different models with different config.

## Step 2, Create the convert.py
using the function `register` to generate the operators.
```python
from mmt import register
import torch.nn as nn
from mobilenetv3 import (conv_3x3_bn,
                         InvertedResidual,
                         conv_1x1_bn,
                         h_swish)
fp = "./mbv3_ops"
reg = lambda ops, **kwargs: register(ops, fp, **kwargs)

reg(conv_3x3_bn,
    inp=[3],
    oup=[16],
    stride=[2],
    input_shape=[[1, 3, 224, 224]])
...
```
It is suggestive to  rewrite the function to add the `fp` to the function
and all operators will be placed in one folder.

Then run the following command:
```
python convert.py
```
## Step 3, Export and Convert your models
Here we use the function `export_models` to save model and convert it to the 
`mnn' format. You could create models with different configs, and save these models
in one folder.
```python
for i in range(200):
    cfg_ = generate_cfg(cfgs)  # generate config
    net = MobileNetV3(cfg_, id=i, mode="small")  # create a model
    export_models(net, [1, 3, 224, 224], "mbv3")
    # save this model at "./mbv3" and convert it to .mnn
```
Models as `torch` and `mnn` format will all be placed in one folder like:
```
MobileNetV31421-3-224-224.mnn       MobileNetV31851-3-224-224meta.json  MobileNetV353.pth                   MobileNetV3961-3-224-224.mnn
MobileNetV3142.pth                  MobileNetV31851-3-224-224.mnn       MobileNetV3541-3-224-224meta.json   MobileNetV396.pth
MobileNetV31431-3-224-224meta.json  MobileNetV3185.pth                  MobileNetV3541-3-224-224.mnn        MobileNetV3971-3-224-224meta.json
MobileNetV31431-3-224-224.mnn       MobileNetV31861-3-224-224meta.json  MobileNetV354.pth                   MobileNetV3971-3-224-224.mnn
MobileNetV3143.pth                  MobileNetV31861-3-224-224.mnn       MobileNetV355.pth                   MobileNetV397.pth
MobileNetV3144.pth                  MobileNetV3186.pth                  MobileNetV3561-3-224-224meta.json   MobileNetV3981-3-224-224meta.json
MobileNetV31451-3-224-224meta.json  MobileNetV31871-3-224-224meta.json  MobileNetV3561-3-224-224.mnn        MobileNetV3981-3-224-224.mnn
```
Running `generate models.py` will generate two folders with a small number of models (for training) and a larger number of models (for validation).
## Step 4, Measure the latency of operators and models
Just run these code at the inference device:
```python
from mmt.meter import meter_ops, meter_models
meter_ops("mbv3_ops")
meter_models("mbv3_train")
meter_models("mbv3_val")
```
Running `measure.py`will measure the latency of operators and models.
## Step 5, Validate the error of prediction
Just run these code at your server:
```python
from mmt.converter import validation
from mmt.predictor import latency_predictor
validation("mbv3_train", "mbv3_ops/meta_latency.pkl", save_path="train_error.csv")
lp = latency_predictor("mbv3_ops", "train_error.csv")
validation("mbv3", "mbv3_ops/meta_latency.pkl", save_path="gp_error.csv", lp=lp)
validation("mbv3", "mbv3_ops/meta_latency.pkl", save_path="error.csv")
```
Verify the error of the training set and save the results to `train_error.csv`, then train the machine learning model to build the delay predictor `lp`, 
and then test the error of the direct prediction delay and the predictor based on the machine learning method on the validation set.