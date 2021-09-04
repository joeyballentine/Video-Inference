from utils.model_classes.base_model import BaseModel
from utils.architectures.TecoGAN_arch import FRNet as TecoGAN

import numpy as np
import torch
import utils.common as util
from collections import OrderedDict


class TecoGanModel(BaseModel):
    def __init__(self, scale=4, nf=64, nb=10, degradation='BI', device=None):
        super(TecoGanModel, self).__init__()
        self.device = device

        self.model = TecoGAN(nf=nf, nb=nb, degradation=degradation).eval()
        self.only_y = False
        self.num_frames = None
        self.num_padding = 0
        self.degradation = degradation
        self.scale = scale

    def load_state_dict(self, state_dict):
        # The BD model from the TecoGAN-pytorch repo has extra keys it doesn't need for some reason
        if 'upsample_func.kernels' in state_dict.keys():
            state_dict.pop('upsample_func.kernels')
        if 'srnet.upsample_func.kernels' in state_dict.keys():
            state_dict.pop('srnet.upsample_func.kernels')
        self.model.load_state_dict(state_dict)        

    def set_io(self, io):
        self.io = io
        self.num_frames = len(io)
    
    def get_frames(self, idx):
        img = self.io[idx]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        # First pass
        if idx == self.num_padding:
            # initialize these variables on first pass
            self.lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(self.device)
            self.hr_prev = torch.zeros(
            1, c, self.scale * h, self.scale * w, dtype=torch.float32).to(self.device)
        return [img]

    def inference(self, LR_img, args):
        LR_img = LR_img[0]
        h_LR, w_LR, c = LR_img.shape
        # list -> numpy # input: list (contatin numpy: [H,W,C])
        LR = [np.asarray(LR_img)]
        LR = np.asarray(LR)  # numpy, [T,H,W,C]
        LR = LR.transpose(1, 2, 3, 0).reshape(
            h_LR, w_LR, -1)  # numpy, [Hl',Wl',CT]
        LR = util.np2tensor(LR, bgr2rgb=True, add_batch=False)
        LR = LR.view(c, 1, h_LR, w_LR)  # Tensor, [C,T,H,W]
        LR = LR.transpose(0, 1)  # Tensor, [T,C,H,W]
        if args.fp16:
            LR = LR.half()
        # LR = LR.unsqueeze(0)
        lr_data = LR

        # setup params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        self.model.to(self.device)

        with torch.no_grad():
            lr_curr = lr_data.to(self.device)
            hr_curr = self.model.forward(lr_curr, self.lr_prev, self.hr_prev)
            self.lr_prev, self.hr_prev = lr_curr, hr_curr
            sr_img = util.tensor2np(hr_curr)

        self.io.save_frames(sr_img)