import os
import random
from PIL import  Image
import numpy as np

match_list = []

unmatch_list = []

class_list = []
COMBINE_PATH = "D:\\Files\\arcface-pytorch\\ROF\\combined"

MASK_PATH = 'D:\\Files\\Dissertation\\ROF\\masked'

NEUTRAL_PATH = 'D:\\Files\\Dissertation\\ROF\\neutral'

GLASSES_PATH = 'D:\\Files\\Dissertation\\ROF\\sunglasses'

LFW_PATH = '/lfw'

MLFW_PATH = "D:\\Files\\arcface-pytorch\\mlfw_dataset\\mlfw_aligned_dir"

def generate_match_list(input_path):
    for folder in os.listdir(input_path):
        imgs_path = os.listdir(os.path.join(input_path, folder))
        path_len = len(imgs_path)
        if not path_len:
            continue

        random.shuffle(imgs_path)
        for img_path in imgs_path:
            rand_idx = random.randint(0, path_len - 1)
            temp_list = [folder, img_path, imgs_path[rand_idx]]
            match_list.append(temp_list)

    return match_list


def get_rand_class(idx, class_num):
    rand_cls_idx = random.randint(0, class_num - 1)
    if rand_cls_idx == idx and idx == 0:
        rand_cls_idx += 1
    elif rand_cls_idx == idx and idx == class_num - 1:
        rand_cls_idx -= 1
    elif rand_cls_idx == idx:
        rand_cls_idx += 1
    else:
        pass
    return rand_cls_idx


def generate_unmatch_list(input_path):
    classes = os.listdir(input_path)
    class_num = len(classes)
    for idx, folder in enumerate(classes):

        imgs_path = os.listdir(os.path.join(input_path, folder))
        path_len = len(imgs_path)
        if not path_len:
            continue

        random.shuffle(imgs_path)
        for img_path in imgs_path:
            rand_cls_idx = get_rand_class(idx, class_num)
            another_cls = classes[rand_cls_idx]
            another_imgs = os.listdir(os.path.join(input_path, another_cls))
            if not another_imgs:
                continue
            rand_img = another_imgs[random.randint(0, len(another_imgs) - 1)]

            temp_list = [folder, img_path, another_cls, rand_img]
            unmatch_list.append(temp_list)
    return unmatch_list


def CEF(path):
    """
    CLean empty files, 清理空文件夹和空文件
    :param path: 文件路径，检查此文件路径下的子文件
    :return: None
    """
    files = os.listdir(path)  # 获取路径下的子文件(夹)列表
    for file in files:
        print ('Traversal at', file)

        if not os.listdir(os.path.join(path, file)): # 如果子文件为空
            os.rmdir(os.path.join(path, file))  # 删除这个空文件夹
            print("remove", file)
    print (path, 'Dispose over!')



if __name__ == "__main__":

    # CEF(GLASSES_PATH)
    unmatch_list = generate_unmatch_list(COMBINE_PATH)
    match_list = generate_match_list(COMBINE_PATH)
    combined_list = unmatch_list + match_list
    random.shuffle(combined_list)
    f = open("D:\\Files\\Dissertation\\ROF\\Combine_pairs.txt", "w")
    for i in combined_list:
        f.write(str(i).replace('[', '').replace(']', '').replace(',', '').replace('\'', '').replace('\'', '') + "\n")
    f.close()

    # for idx, folder in enumerate(os.listdir(GLASSES_PATH)):
    #     imgs_path = os.listdir(os.path.join(GLASSES_PATH, folder))
    #     for idx2, img in enumerate(imgs_path):
    #         path = os.path.join(GLASSES_PATH, folder, img)
    #         img_c = Image.open(path)
    #         if abs(img_c.size[0] / img_c.size[1]) > 1.5 or img_c.size[0] / img_c.size[1] < 0.68:
    #             print(folder)
    #             print(idx2)


