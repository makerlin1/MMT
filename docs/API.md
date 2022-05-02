API Document
---
## core
### core.generate_ops_list(path, fp)
**Description**: Generate operators in mnn format according to the specified YAML file.
### core.convert2mnn(model, input_shape, path, verbose=False):
**Description**: Convert pytorch model to mnn format.

## utils
### utils.parser_model(module, ops_list):
**Description**: Parse the operators in the model and return the corresponding dictionary.
```python
ops_list = [MLP, nn.Linear, nn.ReLU]  # MLP is designed module by yourself
arch2ops = parser_model(model, ops_list, verbose=True)
```
### utils.parse_yaml(fpath)
**Description**: Parse the YAML file and return all operators.
```python
parse_yaml("docs/demo.yaml")
```

    