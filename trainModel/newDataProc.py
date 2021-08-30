import os

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


from PIL import Image
import torch

class RPSdataset():
    def __init__(self,data_dir,transform=None):
        self.dataDir = data_dir
        self.trans = transform

    def allData(self,batchSize):
        image_datasets = {x:datasets.ImageFolder(os.path.join(self.dataDir,x),
                                                 self.trans[x] if self.trans is not None else None)
                          for x in ['train','val',"test"]}
        dataLoaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batchSize,shuffle=True)
                       for x in ['train','val',"test"]}
        # datasets_size = {x:len(image_datasets[x]) for x in ['train','val']}

        # class_names = image_datasets['train'].classes
        return  dataLoaders

if __name__ == '__main__':
    dir = "./data/"
    dataLoaders = RPSdataset(dir,2,transform=None).allData()
    train,val = dataLoaders["train"],dataLoaders["val"]
    # print(len(train))
    # print(len(val))
    # print(len(train.dataset))
    # print(len(val.dataset))