import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from nets.arcface import Arcface
from matplotlib import pyplot as plt
from PIL import Image

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)



def img_process(path):
    img = np.array(Image.open(path))
    cv2.resize(img, (112, 112))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


if __name__ == "__main__":
    # --------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # --------------------------------------#
    cuda = True
    # --------------------------------------#
    #   主干特征提取网络的选择
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    # --------------------------------------#
    backbone = "convNext_cbam"
    # --------------------------------------#
    #   输入图像大小
    # --------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    model_path = "result/conv-cbam-webocc-lfw/ep041-loss18.882-val_loss19.173.pth"
    # --------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    # --------------------------------------#
    lfw_dir_path = "mlfw_dataset/mlfw_aligned_dir/"
    lfw_pairs_path = "mlfw_dataset/mlfw_pairs.txt"
    # --------------------------------------#
    #   评估的批次大小和记录间隔
    # --------------------------------------#
    batch_size = 64
    log_interval = 1
    # --------------------------------------#
    #   ROC图的保存路径
    # --------------------------------------#
    png_save_path = "model_data/roc_test.png"

    # test_loader = torch.utils.data.DataLoader(
    #     TestDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size,
    #     shuffle=False, drop_last=False)

    path_ori = "mlfw_dataset/mlfw_aligned_dir/Ahmed_Chalabi/Ahmed_Chalabi_0004_0001.jpg"
    path_mask = "mlfw_dataset/mlfw_aligned_dir/Ahmed_Chalabi/Ahmed_Chalabi_0001_0001.jpg"

    ori_img, ori_img_float, ori_img_tensor = img_process(path_ori)
    mask_img, mask_img_float, mask_img_tensor = img_process(path_mask)

    model = Arcface(backbone=backbone, mode="predict")


    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model = model.eval()

    ori_out = model(ori_img_tensor)[0, :]
    mask_out = model(mask_img_tensor)[0, :]
    ori_tar = [SimilarityToConceptTarget(ori_out)]
    mask_tar = [SimilarityToConceptTarget(mask_out)]
    target_layers = [model.arcface.stages[3][2]]
    print(model.arcface)
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=False) as cam:
        car_grayscale_cam = cam(input_tensor=ori_img_tensor,
                                targets=mask_tar)[0, :]
    car_cam_image = show_cam_on_image(mask_img_float, car_grayscale_cam, use_rgb=True)
    plt.imshow(car_cam_image)
    plt.show()


# import warnings
#
#
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# from torchvision.models.segmentation import deeplabv3_resnet50
# import torch
# import torch.functional as F
# import numpy as np
# import requests
# import cv2
# import torchvision
# from PIL import Image
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from pytorch_grad_cam import GradCAM
#
#
# # A model wrapper that gets a resnet model and returns the features before the fully connected layer.
# class ResnetFeatureExtractor(torch.nn.Module):
#     def __init__(self, model):
#         super(ResnetFeatureExtractor, self).__init__()
#         self.model = model
#         self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
#
#     def __call__(self, x):
#         return self.feature_extractor(x)[:, :, 0, 0]
#
#
# resnet = torchvision.models.resnet50(pretrained=True)
# resnet.eval()
# model = ResnetFeatureExtractor(resnet)
#
#
# def get_image_from_url(url):
#     """A function that gets a URL of an image,
#     and returns a numpy image and a preprocessed
#     torch tensor ready to pass to the model """
#
#     img = np.array(Image.open(requests.get(url, stream=True).raw))
#     img = cv2.resize(img, (512, 512))
#     rgb_img_float = np.float32(img) / 255
#     input_tensor = preprocess_image(rgb_img_float,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     return img, rgb_img_float, input_tensor
#
# car_img, car_img_float, car_tensor = get_image_from_url("https://www.wallpapersin4k.org/wp-content/uploads/2017/04/Foreign-Cars-Wallpapers-4.jpg")
# cloud_img, cloud_img_float, cloud_tensor = get_image_from_url("https://th.bing.com/th/id/OIP.CmONj_pGCXg9Hq9-OxTD9gHaEo?pid=ImgDet&rs=1")
# car_concept_features = model(car_tensor)[0, :]
# cloud_concept_features = model(cloud_tensor)[0, :]
# image, image_float, input_tensor = get_image_from_url("https://th.bing.com/th/id/R.c65135374de94dea2e2bf8fe0a4818e7?rik=Z75HF5uFr56PAw&pid=ImgRaw&r=0")
# Image.fromarray(image)
#
#
# class SimilarityToConceptTarget:
#     def __init__(self, features):
#         self.features = features
#
#     def __call__(self, model_output):
#         cos = torch.nn.CosineSimilarity(dim=0)
#         return cos(model_output, self.features)
#
#
# target_layers = [resnet.layer4[-1]]
# print(resnet.layer4)
# car_targets = [SimilarityToConceptTarget(car_concept_features)]
# cloud_targets = [SimilarityToConceptTarget(cloud_concept_features)]
#
# # Where is the car in the image
# with GradCAM(model=model,
#              target_layers=target_layers,
#              use_cuda=False) as cam:
#     car_grayscale_cam = cam(input_tensor=input_tensor,
#                             targets=car_targets)[0, :]
# car_cam_image = show_cam_on_image(image_float, car_grayscale_cam, use_rgb=True)
# plt.imshow(car_cam_image)
# plt.show()