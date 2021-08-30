from torchvision import transforms

from newDataProc import RPSdataset
from modelArch import getModel
import torch
from tqdm import tqdm

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

def metric_batch(outPut,target):
  pred = outPut.argmax(dim=1,keepdim=True)
  corrects = pred.eq(target.view_as(pred)).sum().item()
  return corrects


def loss_batch(loss_func,output,target,opt=None):
  loss = loss_func(output,target)
  metric_b = metric_batch(output,target)
  if opt is not None:
    opt.zero_grad()
    loss.backward()
    opt.step()

  return loss.item(),metric_b


def loss_epoch(model,lossfunc,dataLds,opt=None,sanity_check=False):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataLds.dataset)

    for img,lbl in tqdm(dataLds):
        img,lbl = img.to(device),lbl.to(device)

        outModel = model(img)
        loss_b,acc_b = loss_batch(lossfunc,outModel,lbl,opt)
        running_loss += loss_b

        if acc_b is not None:
            running_metric += acc_b
        if sanity_check:
            break
    epoch_loss = running_loss / float(len_data)
    epoch_metric = running_metric / float(len_data)

    return  epoch_loss,epoch_metric


def train_val():
    transes = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.RandomRotation(80),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            # transforms.Normalize(mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
            #                      std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194])
            transforms.Normalize(mean=[0.8364790148158947, 0.8068061890300091, 0.7972531800702287],
                                 std=[0.24796623184980196, 0.28685850871346746, 0.30011578943401435])

        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            # transforms.Normalize(mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
            #                      std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194])
            transforms.Normalize(mean=[0.8364790148158947, 0.8068061890300091, 0.7972531800702287],
                                 std=[0.24796623184980196, 0.28685850871346746, 0.30011578943401435])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,))
            # transforms.Normalize(mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
            #                      std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194])
            transforms.Normalize(mean=[0.8364790148158947, 0.8068061890300091, 0.7972531800702287],
                                 std=[0.24796623184980196, 0.28685850871346746, 0.30011578943401435])
        ])
    }

    model = getModel().to(device)

    # torch.nn.CrossEntropyLoss()
    lossFunc = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataDir = "./data"
    batchSize = 4
    dataLoaders = RPSdataset(dataDir, transes).allData(batchSize)
    trainLd, valLd = dataLoaders["train"], dataLoaders["val"]

    print(f"Classes---\n{trainLd.dataset.classes}")
    num_epochs = 20
    # exit()

    # best_model = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for e in range(1,num_epochs+1):
        model.train()
        train_loss,train_acc = loss_epoch(model,lossFunc,trainLd,optimizer,sanity_check=False)

        model.eval()
        val_loss,val_acc = loss_epoch(model,lossFunc,valLd,sanity_check=False)

        print(f"\nEpoch {e}/{num_epochs} "
              f" TrainLoss: {train_loss:.3f} TrainAcc: {train_acc:.3f}"
              f" ValLoss: {val_loss:.3f} valAcc: {val_acc:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            # best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),"best_model.pth")
            print("Model Saved")

if __name__ == '__main__':
    train_val()