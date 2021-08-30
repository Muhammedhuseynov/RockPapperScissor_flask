from PIL import Image


img = Image.open("../robotHand_scissors.jpg")
img = img.resize((500,500))

img.save("robotHand_scissors.jpg")