import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
import streamlit as st
import concurrent.futures
import tempfile
import importlib.util
import yaml
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
from LAM.size import imresize

import cv2
import yaml as yaml_module
import utils
from LPIPS.lp import lp

class ModelData:
    def __init__(self):
        self.MODEL_DIR = os.path.join('LAM', 'ModelZoo', 'models')
        self.NN_LIST = ['RCAN', 'CARN', 'RRDBNet', 'SAN', 'EDSR', 'HAT', 'SWINIR']
        self.MODEL_LIST = {
            'RCAN': {'Base': 'RCAN.pt'},
            'CARN': {'Base': 'CARN_7400.pth'},
            'RRDBNet': {'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth'},
            'SAN': {'Base': 'SAN_BI4X.pt'},
            'EDSR': {'Base': 'EDSR-64-16_15000.pth'},
            'HAT': {'Base': 'HAT_SRx4_ImageNet-pretrain.pth'},
            'SWINIR': {'Base': "SwinIR.pth"}

        }
        self.metrics = {
            'RCAN': {'psnr': 32.63, 'ssim': 0.9002, 'lpips': 0.1692},
            'CARN': {'psnr': 32.13, 'ssim': 0.8937, 'lpips': 0.1792},
            'RRDBNet': {'psnr': 32.60, 'ssim': 0.9002, 'lpips': 0.1698},
            'SAN': {'psnr': 32.64, 'ssim': 0.9003, 'lpips': 0.1689},
            'EDSR': {'psnr': 32.46, 'ssim': 0.8968, 'lpips': 0.1725},
            'HAT': {'psnr': 33.18, 'ssim': 0.9037, 'lpips': 0.1618},
            'SWINIR': {'psnr': 32.72, 'ssim': 0.9021, 'lpips': 0.1681}
        }

    def proc(self, filename):
        try:
            # 处理图像的代码
            tar, prd = filename
            tar_img = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
            prd_img = cv2.cvtColor(prd, cv2.COLOR_BGR2RGB)
            PSNR = utils.calculate_psnr(tar_img, prd_img, 0, test_y_channel=True)
            SSIM = utils.calculate_ssim(tar_img, prd_img, 0, test_y_channel=True)
            return PSNR, SSIM

        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"Error processing {filename}: {e}")
            return None

    def update(self, weight, yaml_string, arch):
        weight = weight
        yaml_file = yaml_string
        arch_file = arch

        x = yaml.safe_load(yaml_file)
        s = x['network_g'].pop('type')

        self.NN_LIST.append(s)
        model_list = self.NN_LIST

        file_name = weight.name
        self.MODEL_LIST[s] = {
            'Base': file_name
        }
        model_pth_update = self.MODEL_LIST

        arch_name = arch_file.name
        with tempfile.TemporaryDirectory() as tmpdirname:
            arch_path = os.path.join(tmpdirname, arch_name)
            with open(arch_path, 'wb') as tmpfile:
                tmpfile.write(arch_file.read())

            spec = importlib.util.spec_from_file_location(arch_name, arch_path)
            arch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arch)

            model_class = getattr(arch, s)
            model = model_class(**x['network_g'])

            weight_path = os.path.join(tmpdirname, weight.name)
            with open(weight_path, 'wb') as tmpfile:
                tmpfile.write(weight.read())

            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['params'])
            model.cuda()
            model = nn.DataParallel(model)
            model.eval()

            # 计算psnr, ssim, lpip并加进去
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            factor = 8
            factor_test = 4

            file_path = os.path.join('image', 'Set5', 'original')
            files = natsorted(glob(os.path.join(file_path, '*.png')))
            restored_images = []
            original_images = []
            with torch.no_grad():
                for file_ in tqdm(files):  # 输入图像的预处理
                    image = cv2.imread(file_)  # 使用 OpenCV 读取图像文件
                    original_images.append(image)

            with torch.no_grad():
                for file_ in tqdm(files):  # 输入图像的预处理
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()  # 释放GPU内存
                    img = utils.load_img(file_)  # 转为RGB
                    img = img.astype(np.float32) / 255.
                    h_old, w_old, _ = img.shape

                    np.random.seed(seed=0)  # for reproducibility
                    img = imresize(img, scalar_scale=1 / factor_test)  # 图像进行缩放以生成低质量（LQ）图像，然后再将其还原回原始尺寸，用于模拟低分辨率输入
                    img = imresize(img, scalar_scale=factor_test)

                    img = torch.from_numpy(img).permute(2, 0,
                                                        1)  # 图像转换为PyTorch张量，将原始张量的第0维（通道维）、第1维（行维）和第2维（列维）重新排列为新的顺序
                    input_ = img.unsqueeze(0).cuda()  # 扩展维度0意味着在最前面添加一个新的维度，从而将其变成形状为 [1, 通道, 行, 列] 的张量,并移动到GPU上

                    h_pad = (h_old // factor + 1) * factor - h_old
                    w_pad = (w_old // factor + 1) * factor - w_old
                    input_ = torch.cat([input_, torch.flip(input_, [2])], 2)[:, :, :h_old + h_pad, :]
                    input_ = torch.cat([input_, torch.flip(input_, [3])], 3)[:, :, :, :w_old + w_pad]
                    input_ = input_.type(torch.cuda.FloatTensor)

                    model = model.to(device)

                    restored = model(input_)

                    # 恢复后图像保存
                    restored = restored[:, :, :h_old, :w_old]
                    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                    res = img_as_ubyte(restored)
                    restored_images.append(cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

                # 计算PSNR, SSIM, LPIPS

                psnr, ssim = [], []
                inputs = list(zip(original_images, restored_images))
                for input_data in inputs:
                    PSNR_SSIM = self.proc(input_data)
                    # if PSNR_SSIM is not None, which means proc function executed without catching any exception
                    if PSNR_SSIM:
                        psnr.append(PSNR_SSIM[0])
                        ssim.append(PSNR_SSIM[1])

                average_psnr = round(sum(psnr) / len(psnr), 2)
                average_ssim = round(sum(ssim) / len(ssim), 4)
                average_lpips = lp(original_images, restored_images)

                self.metrics[s] = {
                    'psnr': average_psnr,
                    'ssim': average_ssim,
                    'lpips': average_lpips
                }
                metrics_update = self.metrics

                return model_list, model_pth_update, metrics_update
