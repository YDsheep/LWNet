import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.WLNet import WLNet
from data import test_dataset
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='datasets/RGBD_for_test/',help='test dataset path for nsi dataset')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = WLNet()
model.load_state_dict(torch.load('save/NSI/MambaSOD_epoch_best.pth')) # mambaSOD.pth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input1 = torch.randn(1, 3, 384, 384)
input1 = input1.to(device)
input2 = torch.randn(1, 1, 384, 384)
input2 = input2.to(device)
flops, _ = profile(model.cuda(), inputs=(input1,input2,))
print('flops:{}'.format(flops))


model.cuda()
model.eval()


def save_tensor_as_image(tensor, save_dir, name, n):
    # 确保name包含合法扩展名
    valid_ext = ['.jpg', '.png', '.jpeg', '.bmp']
    if not any(name.lower().endswith(ext) for ext in valid_ext):
        name += '.jpg'  # 默认添加.jpg

    # 生成绝对路径
    save_path = os.path.join(save_dir, name)

    # 转换张量为OpenCV格式
    numpy_array = tensor.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
    if numpy_array.dtype == np.float32:
        numpy_array = (numpy_array * 255).clip(0, 255).astype(np.uint8)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    cv2.imwrite(save_path+n, numpy_array)

# test
# test_datasets = ['NJU2K', 'NLPR', 'STERE', 'DES', 'SIP', 'DUT', 'LFSD', 'ReDWeb-S']
test_datasets = ['NJU2K']
for dataset in test_datasets:
    save_path = 'output/NJU2K/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        res,_,_,_,_ = model(image, depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
