#single img predtictor
import torch
from PIL import Image
from torchvision import transforms
from modelArch import getModel
import requests
from io import BytesIO
def predict(img_dir,classes):
    #==== for from online img
    # url  = img_dir
    # response = requests.get(url)
    # print("Loaded------")
    # img = Image.open(BytesIO(response.content()))

    img = Image.open(img_dir)

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8115210741605272, 0.8214176078950116, 0.8501684625825634],
            std=[0.30353752557208274, 0.28861879092499754, 0.24626602132185194]
        )
    ])
    img = trans(img)

    model_name = "./best_model.pth"
    model = getModel().to("cuda")
    model.load_state_dict(torch.load(model_name))

    model.eval()


    outModel = model(img.unsqueeze(0).to("cuda"))
    pred = outModel.argmax(dim=1, keepdim=True)
    # print(pred)
    # print(classes[pred.item()])

    return classes[pred.item()]

if __name__ == '__main__':
    img_dir = "imgs/im.jpg"

    print(predict(img_dir))
