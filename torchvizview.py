import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchviz import make_dot

def build_model(pretrained=True, fine_tune=True, num_classes=7):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = EfficientNet.from_pretrained('efficientnet-b0')
    else:
        print('[INFO]: Not loading pre-trained weights')
        model = EfficientNet.from_name('efficientnet-b0')

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=num_classes)
    return model

model = build_model()
x = torch.randn(1, 3, 224, 224)  # Assuming input size is (batch_size, channels, height, width)
vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
vis_graph.view()


