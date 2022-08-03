import os
import random

PATH = "D:\\Files\\arcface-pytorch\\ROF\\combined"


# rename code
# for name in name_list:
#     folder_path = os.path.join(PATH,name)
#     imgs_list = os.listdir(folder_path)
#     for img_name in imgs_list:
#         if img_name.split("_")[0] != "glasses" and img_name.split("_")[0] != "masked":
#             g = img_name.split("_")[0]
#             img_path = os.path.join(folder_path,img_name)
#             os.rename(img_path,os.path.join(folder_path,"neutral_"+img_name))
# print("rename sucuccessssss")

def match_pairs():
    name_list = os.listdir(PATH)
    paris = []
    for name in name_list:
        folder_path = os.path.join(PATH, name)
        imgs_list = os.listdir(folder_path)
        neutral_list = []
        occ_list = []
        for img_name in imgs_list:
            if  img_name.split("_")[0] == "neutral":
                neutral_list.append(img_name)
            elif img_name.split("_")[0] == "masked":
                occ_list.append(img_name)
            else:
                continue

        if len(occ_list) == 0:
            continue
        count = 0
        for neutral_name in neutral_list:
            if count == 10:
                count = 0
                break

            rand_num = random.randint(0, len(occ_list) - 1)
            paris.append([name, neutral_name,occ_list[rand_num]])
            count += 1
    return paris

def gen_unmatch_folder(idx, name_list):
    rand_folder_num = random.randint(0, len(name_list)-1)
    while rand_folder_num == idx:
        rand_folder_num = random.randint(0, len(name_list)-1)
    occ_folder_path = os.path.join(PATH, name_list[rand_folder_num])
    occ_imgs_list = os.listdir(occ_folder_path)
    occ_list = []
    # print(name_list[rand_folder_num])
    for occ_img_name in occ_imgs_list:
        if occ_img_name.split("_")[0] == "masked":
            occ_list.append(occ_img_name)
    return occ_list, str(name_list[rand_folder_num])

def unmatch_pairs():
    name_list = os.listdir(PATH)
    unmatch_pairs = []
    for idx, name in enumerate(name_list):
        count = 0
        neutral_list = []
        folder_path = os.path.join(PATH, name)
        imgs_list = os.listdir(folder_path)
        for img_name in imgs_list:
            if img_name.split("_")[0] == "neutral":
                neutral_list.append(img_name)
        if len(neutral_list) == 0:
            continue
        for neutral_img in neutral_list:
            if count == 10:
                count = 0
                break
            occ_list,occ_name = gen_unmatch_folder(idx,name_list)
            while len(occ_list) == 0:
                occ_list,occ_name = gen_unmatch_folder(idx,name_list)

            rand_occimg_num = random.randint(0,len(occ_list)-1)
            unmatch_pairs.append([name,neutral_img,occ_name,occ_list[rand_occimg_num]])
            count += 1
    return unmatch_pairs
    # print(unmatch_pairs)





if __name__ == "__main__":
    matchpairs = match_pairs()
    unmatchpairs = unmatch_pairs()
    combined_list = matchpairs + unmatchpairs
    random.shuffle(combined_list)
    f = open("D:\\Files\\arcface-pytorch\\ROF\\n-m.txt", "w")
    for i in combined_list:
        f.write(str(i).replace('[', '').replace(']', '').replace(',', '').replace('\'', '').replace('\'', '') + "\n")
    f.close()
