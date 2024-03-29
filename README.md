# Custom-Pretrained-Models-in-Fastai
Shows how to use your own pretrained models in the Fastai library
> Note that this code works with Fastai version 1.0.54
<br>
1. Depending on what format your model is in, do either of:

```python
# If you exported your model using fastai's export function like so: learn.export()
model_func = model_from_export(PATH_TO_MODEL)
```
```python
# If you exported your model using PyTorch's save feature like so: torch.save(model)
model_func = model_from_torch(PATH_TO_MODEL)
```
2. From here on it's pretty straight forward:

```python
custom_cnn_learner = get_custom_resnet_learner()
learn = custom_cnn_learner(data, model_func, ...usual parameteres---)
```

3. Beyond this point, it's the usual FastAI model training procedures.

*I'd love to hear from you if you find this useful!*
