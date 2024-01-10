# -*- encoding: utf-8 -*-
# @Author:gan
# @File  :grad_cam.py
# @Date  :2030.10.1
# @Function: draw heat map
# example from https://github.com/jacobgil/pytorch-grad-cam


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys
sys.path.append('/home/ubuntu/nanodet')
import argparse

import torch
import cv2 as cv
import numpy as np
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Draw heat map use grad_cam.')
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    parser.add_argument("--img", type=str, help="image path")
    parser.add_argument("--layer-name", type=str, help="which layer to draw heat map")
    args = parser.parse_args()
    return args

def preprocess(img,mean,std):
        mean = np.array(mean, dtype=np.float64).reshape(1, -1)
        stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
        cv.subtract(img, mean, img)
        cv.multiply(img, stdinv, img)
        return img

def main(args):
    load_config(cfg, args.config)
    # model setting
    input_shape = cfg.data.val.input_size
    mean = cfg.data.val.pipeline.normalize[0]
    std = cfg.data.val.pipeline.normalize[1]

    model = build_model(cfg.model)

    params = torch.load(args.model)
    
    model.load_state_dict(params['state_dict'])
    model.eval()
    del model.aux_fpn # delete the aux oart
    del model.aux_head #

    [print(i) for i in model.state_dict().keys()] # print all layers, and select the target layer for the following drawing.

    target_layers = [model.backbone.features[11].conv[0]] # backbone.features.8.conv.0.weight

    # read image
    try:
        raw_image = cv.imread(args.img)
    except FileNotFoundError:
        raise ("image not found!")
    image = cv.imread(args.img)
    # BGR to RGB
    raw_image = raw_image[...,::-1]
    image    = image[...,::-1]

    
    # resize
    image = cv.resize(image, input_shape)
    raw_image = cv.resize(raw_image, input_shape)
    raw_image = np.float32(np.array(raw_image) / 255)
    # normalize
    image = preprocess(image,mean,std)
    # preprocess it
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)


    input_tensor = torch.tensor(image)# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = EigenCAM(model=model, target_layers=target_layers)
    # model(input_tensor)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(raw_image, grayscale_cam, use_rgb=True)
    #visualization = np.array(visualization,dtype='uint8')
    cv.imwrite('./cam.jpg',visualization[...,::-1])



if __name__ == '__main__':
    args = parse_args()
    main(args)





