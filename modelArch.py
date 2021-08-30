import torch
from torchvision.models import resnet50,mobilenet_v2


def getModel():
    model = mobilenet_v2(pretrained=True)
    

    model.classifier[1] = torch.nn.Linear(model.last_channel, 4)
   

    return model