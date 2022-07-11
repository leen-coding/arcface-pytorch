import os
import shutil

masked_path = "masked"
neutral_path = "neutral"
sunglasses_path = "sunglasses"
combined_path = "combined"
if not os.path.exists(combined_path):
    os.mkdir(combined_path)


masked_list = os.listdir(masked_path)
sunglasses_list = os.listdir(sunglasses_path)
neutral_list = os.listdir(neutral_path)

for neutral_name in neutral_list:
    imgs_path = os.path.join(neutral_path, neutral_name)
    each_folder = os.path.join(combined_path, neutral_name)
    if not os.path.exists(each_folder):
        os.mkdir(each_folder)

    imgs_path_list = os.listdir(imgs_path)
    for img_path in imgs_path_list:
        shutil.copy(os.path.join(imgs_path, img_path), each_folder)

    for masked_name in masked_list:
        name = masked_name.split("_")[0]+"_"+masked_name.split("_")[1]
        if name == neutral_name:
            masked_imgs_path = os.path.join(masked_path, masked_name)
            masked_img_list = os.listdir(masked_imgs_path)
            for mask_img in masked_img_list:
                shutil.copy(os.path.join(masked_imgs_path,mask_img),os.path.join(each_folder, "masked_{}".format(mask_img)))

    for sunglasses_name in sunglasses_list:
        name = sunglasses_name.split("_")[0]+"_"+sunglasses_name.split("_")[1]
        if name == neutral_name:
            glasses_imgs_path = os.path.join(sunglasses_path, sunglasses_name)
            glasses_img_list = os.listdir(glasses_imgs_path)
            for glasses_img in glasses_img_list:
                shutil.copy(os.path.join(glasses_imgs_path ,glasses_img),os.path.join(each_folder, "glasses_{}".format(glasses_img)))




