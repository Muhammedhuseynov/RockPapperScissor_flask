import matplotlib.pyplot as plt
import numpy as np

def unNormolize(img):
    mean = [0.5,0.5,0.5]
    std  = [0.5,0.5,0.5]
    image = std * img + mean
    image = np.clip(image,0,1)
    return image
def getDicKey(dict,val):
    for k,v in dict.items():
        if v == val:
            return k
    return None

def visualizPredictedRes(**params):
    plt.figure(figsize=(16,5))

    imgs = params["imgs"]
    lenImgs = len(imgs)
    # classes should be dictionary; ex: {'paper': 0, 'rock': 1, 'scissors': 2}
    classes = params["classes"]
    orig_lbls = params["targets"]
    pred_lbls = params["predLbls"]

    for i in range(lenImgs):
        img = imgs[i].cpu().numpy().squeeze().transpose(1,2,0)

        if params["unnormolize"]:
            img = unNormolize(img)
        plt.subplot(1,lenImgs,i + 1)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img)
        if orig_lbls is not None:
            plt.title(f"{getDicKey(classes,pred_lbls[i])} Tr:({getDicKey(classes,orig_lbls[i])})",
                      color=("green" if pred_lbls[i] == orig_lbls[i] else "red"))
    plt.show()