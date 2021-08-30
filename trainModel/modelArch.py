#CNN
import torch
from torchvision.models import resnet50,mobilenet_v2


def getModel():
    model = mobilenet_v2(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # model_inp_feat = model.fc.in_features

    model.classifier[1] = torch.nn.Linear(model.last_channel, 4)
    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(model_inp_feat,256),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.2),
    #     torch.nn.Linear(256,3),
    #     torch.nn.LogSoftmax(dim=1)
    # )

    # model.fc = torch.nn.Linear(in_features=model_inp_feat,out_features=3,bias=True)

    return model




# def getModel():
#     model = resnet50(pretrained=True)
#
#
#     model_inp_feat = model.fc.in_features
#
#
#
#     model.fc = torch.nn.Linear(in_features=model_inp_feat,out_features=3,bias=True)
#
#     return model

if __name__ == '__main__':
    model = getModel()
    print(model)
    # x = torch.randn(4,3,224,224)
    # out = model(x)
    # print(out.data)
    # c,pred = torch.max(out.data,1)
    # print(pred)
    # print(c)
    # print(out)