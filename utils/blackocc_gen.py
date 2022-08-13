import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def gen_occ(PATH,ratio = 0.2):
    new_path = PATH.split(".")[0]+ "_occ.jpg"
    img = Image.open(PATH)
    H = img.size[0]
    W = img.size[1]
    length = int(ratio*min(H, W))
    occ = Image.new('RGB', (length, length))
    new_image = Image.new('RGB', img.size)
    new_image.paste(img)
    randX = random.randint(0, H-length)
    randY = random.randint(0, W-length)
    Image.Image.paste(new_image, occ, (randX, randY))
    # plt.imshow(new_image)
    # plt.show()
    new_image.save(new_path)

PATH = "D:\\Files\\dataset\\lfws\\lfw-factor-9"

folder_list = os.listdir(PATH)

for path in folder_list:
    img_list = os.listdir(os.path.join(PATH, path))
    for img in img_list:
        img_path = os.path.join(PATH,path,img)
        gen_occ(img_path, 0.9)
