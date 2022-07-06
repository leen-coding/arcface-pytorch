import os
import shutil
path = 'D:\\Files\\dataset\\aligned'
list_path = os.listdir(path)
new_path = 'D:\\Files\\dataset\\mlfw_aligned_dir'
if not os.path.exists(new_path):
    os.mkdir('D:\\Files\\dataset\\mlfw_aligned_dir')





for img in list_path:
    img_path = os.path.join(path, img)
    name = img.split("_")
    dir_name = name[0]+"_"+name[1]
    if not os.path.exists(os.path.join(new_path,dir_name)):
        os.mkdir(path=os.path.join(new_path, dir_name))

    cur_dir = os.path.join(new_path, dir_name)
    shutil.copy(img_path, cur_dir)
