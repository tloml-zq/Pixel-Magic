import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
#from restormerarch import Restormer
import argparse
from pdb import set_trace as stx
import numpy as np
from PIL import Image
from Restormer import utils


def get_weights_and_parameters(task, parameters):
    global weights
    if task == '运动去模糊':
        weights = os.path.join('Restormer', 'Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == '单图像散焦去模糊':
        weights = os.path.join('Restormer', 'Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == '图像去雨':
        weights = os.path.abspath('Restormer/Motion_Deblurring/pretrained_models/deraining.pth')
        #weights = os.path.join('Restormer', 'Motion_Deblurring', 'pretrained_models', 'deraining.pth')
    elif task == '高斯彩色去噪':
        weights = os.path.join('Restormer', 'Motion_Deblurring', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == '高斯灰度去噪':
        weights = os.path.join('Restormer', 'Motion_Deblurring', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] =  1
        parameters['out_channels'] =  1
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters


def main(input_img, choice):
    inp_dir = input_img

    if choice == '图像去雨':
        weight = os.path.join('Restormer', 'Motion_Deblurring', 'pretrained_models', 'deraining.pth')
        # weight = 'Restormer/Motion_Deblurring/pretrained_models/deraining.pth'

        ####### Load yaml #######
        yaml_file = os.path.join('Restormer', 'Deraining', 'Options', 'Deraining_Restormer.yml')
        # yaml_file = 'Restormer/Deraining/Options/Deraining_Restormer.yml'
        import yaml

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')

        # load_arch = run_path('Restormer/basicsr/models/archs/restormer_arch.py')
        load_arch = run_path(os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
        model_restoration = load_arch['Restormer'](**x['network_g'])

        checkpoint = torch.load(weight)
        model_restoration.load_state_dict(checkpoint['params'])
        model_restoration.cuda()
        model_restoration = nn.DataParallel(model_restoration)
        model_restoration.eval()

        factor = 8

        with torch.no_grad():
            img = np.float32(utils.load_img(inp_dir)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored_img = cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            restore_img = Image.fromarray(restored_img)


    else:
        # Get model weights and parameters
        parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                      'num_refinement_blocks': 4, 'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False,
                      'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
        weight, parameters = get_weights_and_parameters(choice, parameters)

        load_arch = run_path(os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))

        model = load_arch['Restormer'](**parameters)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint['params'])
        model.eval()

        img_multiple_of = 8

        with torch.no_grad():
            if choice == '高斯灰度去噪':
                img = utils.load_gray_img(inp_dir)
            else:
                img = utils.load_img(inp_dir)

            input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

            # Pad the input if not_multiple_of 8
            height, width = input_.shape[2], input_.shape[3]
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :height, :width]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            if choice == '高斯灰度去噪':
                restored_img = restored
                restored_img = np.squeeze(restored_img)
                restore_img = Image.fromarray(restored_img.astype(np.uint8))
            else:
                restored_img = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                restore_img = Image.fromarray(restored_img)

    return restore_img