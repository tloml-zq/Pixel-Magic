import argparse
import os
import lpips
import numpy as np

def lp(original_images, restored_images):
    loss_fn = lpips.LPIPS(net='alex', version=0.1)
    loss_fn.cuda()

    lpips_distances = []

    for orig_img, rest_img in zip(original_images, restored_images):
        # 图片预处理
        orig_img = orig_img[:,:,::-1]
        rest_img = rest_img[:,:,::-1]

        orig_img_tensor = lpips.im2tensor(orig_img.astype(np.float32))
        rest_img_tensor = lpips.im2tensor(rest_img.astype(np.float32))

        orig_img_tensor = orig_img_tensor.cuda()
        rest_img_tensor = rest_img_tensor.cuda()

        dist = loss_fn.forward(orig_img_tensor, rest_img_tensor)
        lpips_distances.append(dist.item())

    average_lpips = sum(lpips_distances) / len(lpips_distances)
    return round(average_lpips, 4)