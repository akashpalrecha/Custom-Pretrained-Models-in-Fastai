from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
from fastai.basic_train import load_learner
from fastai.vision.learner import cnn_learner
from fastai.vision.learner import _resnet_split

def model_func(f):
    """ Decorator to create functions that return models with an API compatible with Fastai"""
    def func(pretrained=True):
        # The pretrained parameter is only a placeholder to make it compatible with fastai
        # Because in this case, we will always have a pretrained model anyway
        return f()

def model_from_export(PATH: Path):
    """ Get fastai compatible model function from model file at PATH
        The model file has to be created using `learn.export()` in fastai. (Only ResNets supported)
    """
    model = load_learner(PATH).model
    m1, m2 = list(m.children())
    @model_func
    def model():
        return nn.Sequential(*m1.children(), m2[0], m2[1:])
    return model

def model_from_torch(PATH):
    """ Get fastai compatible model function from model file at PATH
        The model file has to be created using 'torch.save(model)`
    """
    m = torch.load(str(PATH))
    @model_func
    def model():
        return m
    return model

def get_custom_resnet_learner():
    """ Returns function to replace `cnn_learner` when using custom pretrained ResNets"""
    return partial(cnn_learner, split_on=_resnet_split, cut=-2)

def print_instructions():
    print("Note that this code works with Fastai version 1.0.54")
    print("Step 1: Depending on what format your model is in, do either of:")
    print("model_func = model_from_export(PATH_TO_MODEL)")
    print("model_func = model_from_torch(PATH_TO_MODEL)")
    print("Step 2:")
    print("custom_cnn_learner = get_custom_resnet_learner()")
    print("learn = custom_cnn_learner(data, model_func, ...usual parameters...)")