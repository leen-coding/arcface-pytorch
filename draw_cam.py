import torch
import torch.backends.cudnn as cudnn
from torchcam.methods import SmoothGradCAMpp
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from nets.arcface import Arcface
from utils.dataloader import TestDataset
from utils.utils_metrics import test
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
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
    backbone = "mobilefacenet"
    # --------------------------------------#
    #   输入图像大小
    # --------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    model_path = "result/mobileface-webocc-lfw/ep034-loss4.674-val_loss5.314.pth"
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

    test_loader = torch.utils.data.DataLoader(
        TestDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size,
        shuffle=False, drop_last=False)

    model = Arcface(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model = model.eval()
    cam_extractor = SmoothGradCAMpp(model)
    times = 0
    img_path = "mlfw_dataset/mlfw_aligned_dir/Aaron_Eckhart/Aaron_Eckhart_0001_0000.jpg"

    img = read_image(img_path)
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (112, 112)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out)
    import matplotlib.pyplot as plt
    from torchcam.utils import overlay_mask

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result);
    plt.axis('off');
    plt.tight_layout();
    plt.show()
    # for (data_a, data_p, label) in test_loader:
    #     if times == 1:
    #         break
    #     out_a = model(data_a)
    #     out_p = model(data_p)
    #     times += 1
    #
    #     activation_map = cam_extractor(0, out_a)
        # result = overlay_mask(out_a, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # plt.imshow(result);
        # plt.axis('off');
        # plt.tight_layout();
        # plt.show()