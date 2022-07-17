import os
from PIL import Image
ROF_path = "D:\\Files\\arcface-pytorch\\ROF\\combined"

name_list = os.listdir(ROF_path)

for name in name_list:
    folder_path = os.path.join(ROF_path, name)
    imgs_path_list = os.listdir(folder_path)
    for imgs_path in imgs_path_list:
        img_path = os.path.join(folder_path, imgs_path)
        f = Image.open(img_path)
        if f.size[0] < 50 or f.size[1] < 50:
            print("removing {}".format(img_path))
            f.close()
            os.remove(img_path)

print("remove complete!")