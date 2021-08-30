

import torch

from torchvision import transforms

from newDataProc import RPSdataset

from visualize import visualizPredictedRes
from modelArch import getModel
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

device = "cuda"

# testDir = "./data/test"
# # testDir = "./data/test"
# data = getImgWithPath(testDir)
# # print(data)
#
# classes = os.listdir(testDir)
# classesDict = {clName: i for i, clName in enumerate(classes)}
# print(classesDict)
#
#
# testData = DataPorc(data,classesDict,mode="val")
#
#
# testLd = DataLoader(testData,batch_size=5,shuffle=True)


transes = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size=544, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=512),  # Image net standards
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            transforms.Normalize(mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
                                 std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194])

        ]),
        "val": transforms.Compose([
            transforms.Resize(size=224),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            transforms.Normalize(mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
                                 std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
            # transforms.Normalize(mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
            #                      std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194])
        ])
}


loaders = RPSdataset("./data",transes).allData(4)
testLd = loaders["test"]
# {'paper': 0, 'rock': 1, 'scissors': 2}
# print(testLd.dataset.classes)

classesDict = {clName:i for i,clName in enumerate(testLd.dataset.classes)}
# print(classesDict)
# exit()
model = getModel().to(device)
model_name = "./best_model.pth"

model.load_state_dict(torch.load(model_name))



def accuracy(outModel,lbl):
    # acc = 0
    top_prob,pred_classes = outModel.topk(1,dim=1)
    # print(top_prob)
    # print(top_cl)

    corrects = pred_classes == lbl.view(*pred_classes.shape)

    return  pred_classes,torch.mean(corrects.type(torch.FloatTensor))


with torch.no_grad():
    model.eval()

    # valAcc = 0
    #
    # for imgVal, lblVal in tqdm(testLd):
    #     imgVal, lblVal = imgVal.to(device), lblVal.to(device)
    #
    #     out = model(imgVal)
    #     # print(out.shape)
    #     # out = torch.unsqueeze(out,0)
    #     # print(out.shape)
    #     # print(lblVal.shape)
    #     # exit()
    #
    #
    #     valAcc += accuracy(out, lblVal)
    # print(f"Acc: {valAcc/len(testLd) * 100}%")




    for _ in range(10):
        img, lbl = next(iter(testLd))

        img,lbl = img.to(device), lbl.to(device)
        out = model(img)
        pred_classes,_ = accuracy(out,lbl)
        print(pred_classes)
        print(lbl)
        visualizPredictedRes(imgs=img, targets=lbl.cpu().numpy(),
                             predLbls=pred_classes.cpu().numpy(),classes=classesDict,
                             unnormolize=False)

# 47.84946060180664%