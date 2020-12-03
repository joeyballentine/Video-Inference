from numpy.lib.function_base import interp
from utils.model_classes.base_model import BaseModel
from utils.architectures.RIFE_arch import RIFE
from utils.architectures.RIFE_HD_arch import RIFE as RIFEHD
import utils.common as util

import torch
from torch.nn import functional as F
import numpy as np
import cv2

import warnings

class RIFEModel(BaseModel):
    def __init__(self, device=None):
        super(RIFEModel, self).__init__()
        self.device = device

        self.model = RIFE(device=self.device).eval()
        self.num_frames = 2
        self.num_padding = 0
        self.only_y = False
        self.denoise = False
        self.scale = 1

    def get_frames(self, idx, is_video=False):
        LR_list = []
        for i in range(self.num_frames):
            if idx + i < len(self.data):
                # Read image or select video frame
                LR_img = self.data[idx + i] if is_video else cv2.imread(
                    self.data[idx + i], cv2.IMREAD_COLOR)
                LR_list.append(LR_img)
        return LR_list

    def inference(self, LR_list, args):
        exp=args.exp
        
        # This is to ignore float scale factor warning
        warnings.filterwarnings(action='ignore', category=UserWarning)

        # TODO: Implement chop_forward for RIFE

        imgs = [util.np2tensor(img).to(self.device) for img in LR_list]
        
        if len(LR_list) == 2:
            img0, img1 = imgs
            n, c, h, w = img0.shape
            # TODO: Check if padding is necessary
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            img_list = [img0, img1]
            self.model.to(self.device)
            with torch.no_grad():
                for _ in range(exp):
                    tmp = []
                    for j in range(len(img_list) - 1):
                        mid = self.model(img_list[j], img_list[j + 1], training=False)
                        tmp.append(img_list[j])
                        tmp.append(mid)
                    tmp.append(img1)
                    img_list = tmp
            output = [util.tensor2np(interp[0].detach().cpu()) for interp in img_list][:-1]
        else:
            output = [LR_list[0]]

        return output

class RIFE_HD_Model(RIFEModel):
    def __init__(self, device=None):
        super(RIFE_HD_Model, self).__init__()
        self.device = device

        self.model = RIFEHD(device=self.device).eval()
        # self.num_frames = 2
        # self.num_padding = 0
        # self.only_y = False
        # self.denoise = False
        # self.scale = 1