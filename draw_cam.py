import cv2
# from sklearn import preprocessing
import numpy as np
import torch
import torchvision.transforms.functional
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from nets.arcface import Arcface
from matplotlib import pyplot as plt
from PIL import Image
from utils.utils import resize_image, preprocess_input, cvtColor
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from utils.dataloader import TestDataset


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        test = cos(model_output, self.features)
        return test

    # def __call__(self, model_output):
    #     dists = torch.sqrt(torch.sum((model_output - self.features) ** 2, 0))
    #     return  1-dists


def img_process(path, inputsize):
    # img = np.array(Image.open(path))
    # img = cv2.resize(img, (112, 112))
    # rgb_img_float = np.float32(img) / 255
    # input_tensor = preprocess_image(rgb_img_float,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    img = Image.open(path)  # 160*160*3 int
    img = np.array(resize_image(img, inputsize, letterbox_image=True))  # 112*112*3 int
    rgb_img_float = np.float32(img) / 255
    rgb_img_float_next = preprocess_input(np.array(img, dtype='float32'))  # 112*112*3 float
    input_tensor = torchvision.transforms.functional.to_tensor(rgb_img_float_next).unsqueeze(0)  # 1*3*112*112

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
    backbone1 = "mobilefacenet"
    model_path1 = "result/mobileface-web1o2/ep050-loss7.186-val_loss8.717.pth"
    backbone2 = "mobilefacenet_two_branch_v3"
    model_path2 = "result/mobileface-web1o2-two_branch_v3/ep032-loss6.853-val_loss8.570.pth"


    # mobilefacenet_two_branch_v6
    # "result/mobileface-web1o2-7_branch_cbam_v6/ep049-loss5.901-val_loss7.872.pth"
    # --------------------------------------#
    #   输入图像大小

    # --------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    # --------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    # --------------------------------------#
    lfw_dir_path = "ROF/combined"
    lfw_pairs_path = "ROF/ROFCombine_m_un_pairs_clean.txt"
    # --------------------------------------#
    #   评估的批次大小和记录间隔
    # --------------------------------------#
    batch_size = 1
    log_interval = 1
    # --------------------------------------#
    #   ROC图的保存路径
    # --------------------------------------#
    png_save_path = "model_data/roc_test.png"

    test_loader = torch.utils.data.DataLoader(
        TestDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size,
        shuffle=True, drop_last=False)
    time = 0
    num = 40
    model1 = Arcface(backbone=backbone1, mode="predict")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1.load_state_dict(torch.load(model_path1, map_location=device), strict=False)
    model = model1.eval()
    # model1.arcface.sep
    target_layers1 = [model1.arcface.sep]
    # model1.arcface.stages[3]
    model2 = Arcface(backbone=backbone2, mode="predict")
    model2.load_state_dict(torch.load(model_path2, map_location=device), strict=False)
    model2 = model2.eval()
    print(model2.arcface)
    target_layers2 = [model2.arcface.stage1[-1]]

    for batch in test_loader:
        if time <= num:
            img1, img2, issame = batch
            img1_plt = np.array(torch.permute(torch.squeeze(img1), (1, 2, 0))) * 0.5 + 0.5
            img2_plt = np.array(torch.permute(torch.squeeze(img2), (1, 2, 0))) * 0.5 + 0.5
            img2_numpy = np.array(img2_plt, dtype='float32')
            # img2_numpy = img2_plt.numpy()
            plt.figure(1)
            plt.subplot(1, 4, 1)
            plt.imshow(img1_plt)
            plt.title("ori")
            plt.subplot(1, 4, 2)
            plt.imshow(img2_numpy)
            plt.title("occ")
            ori_out1 = model1(img1)[0, :]

            print(issame)

            ori_np1 = ori_out1.detach().numpy().reshape(-1,1)
            mask_out1 = model1(img2)[0, :]
            mask_np1 = mask_out1.detach().numpy().reshape(-1,1)
            dists1 = torch.sqrt(torch.sum((ori_out1 - mask_out1) ** 2, 0))
            norm_out1 = normalization(ori_np1[:,0])
            norm_mask_out1 = normalization(mask_np1[:,0])
            norm_dist1 = np.sqrt(np.sum((norm_out1-norm_mask_out1)**2,0))
            print("embeddings 1 dis = {}".format(norm_dist1))



            ori_tar1 = [SimilarityToConceptTarget(ori_out1)]
            ori_out2 = model2(img1)[0, :]

            ori_np2 = ori_out2.detach().numpy().reshape(-1,1)
            mask_out2 = model2(img2)[0, :]
            mask_np2 = mask_out2.detach().numpy().reshape(-1,1)
            dists2 = torch.sqrt(torch.sum((ori_out2 - mask_out2) ** 2, 0))
            norm_out2 = normalization(ori_np2[:,0])
            norm_mask_out2 = normalization(mask_np2[:,0])
            norm_dist2 = np.sqrt(np.sum((norm_out2-norm_mask_out2)**2,0))
            print("embeddings 2 dis = {}".format(norm_dist2))

            ori_tar2 = [SimilarityToConceptTarget(ori_out2)]
            with GradCAM(model=model1,
                         target_layers=target_layers1,
                         use_cuda=False) as cam1:
                model1_cam = cam1(input_tensor=img2,
                                        targets=ori_tar1)[0, :]
            model1_img = show_cam_on_image(img2_plt, model1_cam, use_rgb=True)
            plt.subplot(1, 4, 3)
            plt.imshow(model1_img)
            plt.title("cam_model1")
            with GradCAM(model=model2,
                         target_layers=target_layers2,
                         use_cuda=False) as cam2:
                model2_cam = cam2(input_tensor=img2,
                                        targets=ori_tar2)[0, :]
            model2_img = show_cam_on_image(img2_plt, model2_cam, use_rgb=True)
            plt.subplot(1, 4, 4)
            plt.imshow(model2_img)
            plt.title("cam_model2")
            plt.show()
        time += 1

    # # mfr2/BrodyJenner/BrodyJenner_0001.png
    # path_mask = "ROF/combined/bruce_willis/neutral_000006.jpg"
    # path_ori = "ROF/combined/bruce_willis/neutral_000005.jpg"
    # # (input_shape[0], input_shape[1])
    # ori_img, ori_img_float, ori_img_tensor = img_process(path_ori,(input_shape[0], input_shape[1]))
    # mask_img, mask_img_float, mask_img_tensor = img_process(path_mask,(input_shape[0], input_shape[1]))
    #
    #
    #
    # print('Loading weights into state dict...')
    #
    #
    # ori_out = model(ori_img_tensor)[0, :]
    # ori_np = ori_out.detach().numpy().reshape(-1,1)
    # mask_out = model(mask_img_tensor)[0, :]
    # mask_np = mask_out.detach().numpy().reshape(-1,1)
    # dists = torch.sqrt(torch.sum((ori_out - mask_out) ** 2, 0))
    # norm_out = normalization(ori_np[:,0])
    # norm_mask_out = normalization(mask_np[:,0])
    # norm_dist = np.sqrt(np.sum((norm_out-norm_mask_out)**2,0))
    # print("embeddings dis = {}".format(norm_dist))
    # ori_tar = [SimilarityToConceptTarget(ori_out)]
    # mask_tar = [SimilarityToConceptTarget(mask_out)]
    # if backbone =="mobilefacenet" or backbone == "mobilefacenet_cbam":
    #     target_layers = [model.arcface.sep]
    # else:
    #     target_layers = [model.arcface.stages[3]]
    # # model.arcface.stages[3]
    # # model.arcface.sep
    # # print(model.arcface)
    # with GradCAM(model=model,
    #              target_layers=target_layers,
    #              use_cuda=False) as cam:
    #     car_grayscale_cam = cam(input_tensor=mask_img_tensor,
    #                             targets=ori_tar)[0, :]
    # car_cam_image = show_cam_on_image(mask_img_float, car_grayscale_cam, use_rgb=True)
    # plt.imshow(car_cam_image)
    # plt.show()
