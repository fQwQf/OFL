from .resnet_big import *
from .otfusion_model import *
from .lightweight_model import *

def get_train_models(model_name, num_classes, mode, **kwargs):
    if mode == 'unsupervised':
        train_model = SupConResNet(model_name, head=kwargs['head'])
        if kwargs['classifier'] == 'linear':
            classifier = LinearClassifier(model_name, num_classes=num_classes)
        elif kwargs['classifier'] == 'mlp':
            classifier = MLPClassifier(model_name, num_classes=num_classes)
        return train_model, classifier
    elif mode == 'ot':
        model = get_model_for_ot(model_name, n_c=num_classes)
        return model
    elif mode == 'etf':
        model = ETFCEResNet(model_name, num_classes=num_classes)
        return model
    elif mode == 'our':
        if 'mobilenet' in model_name:
            model = LearnableProtoMobileNet(model_name, num_classes=num_classes)
        else:
            model = LearnableProtoResNet(model_name, num_classes=num_classes)

        return model
    else:
        if 'mobilenetv2' in model_name:
            model = SupConMobileNet(model_name, feat_dim=num_classes)
        else:
            model = SupCEResNet(model_name, num_classes=num_classes)
        return model