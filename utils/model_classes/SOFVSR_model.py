from utils.model_classes.base_model import BaseModel
from utils.architectures.SOFVSR_arch import SOFVSR
import utils.common as util
from utils.colors import *

import cv2
import numpy as np
import torch
from torch.autograd import Variable


class SOFVSRModel(BaseModel):
    def __init__(self, only_y=True, num_frames=3, num_channels=320, scale=4, SR_net='sofvsr', img_ch=1, device=None):
        super(SOFVSRModel, self).__init__()
        self.device = device

        self.model = SOFVSR(scale=scale, n_frames=num_frames,
                            channels=num_channels, SR_net=SR_net, img_ch=img_ch).eval()
        self.only_y = only_y
        self.num_frames = num_frames
        self.num_padding = (num_frames - 1) // 2
        self.num_channels = num_channels
        self.scale = scale
        self.SR_net = SR_net
        self.img_ch = img_ch

        self.previous_lr_list = []
    
    def get_frames(self, idx):
        # First pass
        if idx == self.num_padding:
            LR_list = []
            # Load all beginning images on either side of current index
            # E.g. num_frames = 7, from -3 to 3
            for i in range(-self.num_padding, self.num_padding + 1):
                # Read image or select video frame
                LR_img = self.io[idx + i]
                LR_list.append(LR_img)
        # Other passes
        else:
            # Remove beginning frame from cached list
            LR_list = self.previous_lr_list[1:]
            # Load next image or video frame
            new_img = self.io[idx + self.num_padding]
            LR_list.append(new_img)
        # Cache current list for next iter
        self.previous_lr_list = LR_list
        return LR_list

    def inference(self, LR_list, args):
        denoise=args.denoise
        chop_forward=args.chop_forward

        num_padding = (self.num_frames - 1) // 2
        LR_bicubic = None

        if self.only_y:
            # Convert LR_list to grayscale
            gray_lr_list = []
            LR_bicubic = LR_list[num_padding]
            for i in range(len(LR_list)):
                gray_lr = util.bgr2ycbcr(LR_list[i], only_y=True)
                gray_lr = util.fix_img_channels(gray_lr, 1)
                gray_lr_list.append(gray_lr)
            LR_list = gray_lr_list

            # Get the bicubic upscale of the center frame to concatenate for SR
            if denoise:
                LR_bicubic = cv2.blur(LR_bicubic, (3, 3))
            else:
                LR_bicubic = LR_bicubic
            LR_bicubic = util.imresize_np(
                img=LR_bicubic, scale=self.scale)  # bicubic upscale

            LR = np.concatenate((LR_list), axis=2)  # h, w, t
            # Tensor, [CT',H',W'] or [T, H, W]
            LR = util.np2tensor(LR, bgr2rgb=False, add_batch=True)
            if args.fp16:
                LR = LR.half()

            # generate Cr, Cb channels using bicubic interpolation
            LR_bicubic = util.bgr2ycbcr(LR_bicubic, only_y=False)
            LR_bicubic = util.np2tensor(
                LR_bicubic, bgr2rgb=False, add_batch=True)
            if args.fp16:
                LR_bicubic = LR_bicubic.half()
        else:
            # TODO: Figure out why this is necessary
            LR_list = [cv2.cvtColor(LR_img, cv2.COLOR_BGR2RGB) for LR_img in LR_list]

            h_LR, w_LR, c = LR_list[0].shape
            t = self.num_frames
            # list -> numpy # input: list (contatin numpy: [H,W,C])
            LR = [np.asarray(LT) for LT in LR_list]
            LR = np.asarray(LR)  # numpy, [T,H,W,C]
            LR = LR.transpose(1, 2, 3, 0).reshape(
                h_LR, w_LR, -1)  # numpy, [Hl',Wl',CT]

            # Tensor, [CT',H',W'] or [T, H, W]
            LR = util.np2tensor(LR, bgr2rgb=True, add_batch=False)
            LR = LR.view(c, t, h_LR, w_LR)  # Tensor, [C,T,H,W]
            LR = LR.transpose(0, 1)  # Tensor, [T,C,H,W]
            if args.fp16:
                LR = LR.half()
            LR = LR.unsqueeze(0)

            LR_bicubic = []

        if len(LR.size()) == 4:
            b, n_frames, h_lr, w_lr = LR.size()
            LR = LR.view(b, -1, 1, h_lr, w_lr)  # b, t, c, h, w
        # for networks that work with 3 channel images
        elif len(LR.size()) == 5:
            _, n_frames, _, _, _ = LR.size()
            LR = LR  # b, t, c, h, w

        if chop_forward:
            # crop borders to ensure each patch can be divisible by 2
            # TODO: Modify this to expand the image instead, then crop after
            _, _, _, h, w = LR.size()
            h = int(h//16) * 16
            w = int(w//16) * 16
            LR = LR[:, :, :, :h, :w]
            if isinstance(LR_bicubic, torch.Tensor):
                SR_cb = LR_bicubic[:, 1, :h * self.scale, :w * self.scale]
                SR_cr = LR_bicubic[:, 2, :h * self.scale, :w * self.scale]

            SR_y = self.chop_forward(LR, self.model, self.scale).squeeze(0)
            if self.only_y:
                sr_img = ycbcr_to_rgb(torch.stack((SR_y, SR_cb, SR_cr), -3))
            else:
                sr_img = SR_y
        else:

            with torch.no_grad():
                self.model.to(self.device)
                _, _, _, fake_H = self.model(LR.to(self.device))

            SR = fake_H.detach()[0].float().cpu()
            if self.only_y:
                SR_cb = LR_bicubic[:, 1, :, :]
                SR_cr = LR_bicubic[:, 2, :, :]
                sr_img = ycbcr_to_rgb(torch.stack((SR, SR_cb, SR_cr), -3))
            else:
                sr_img = SR

        sr_img = util.tensor2np(sr_img)  # uint8
        self.io.save_frames(sr_img)

    def chop_forward(self, x, model, scale, shave=16, min_size=5000, nGPUs=1):
        # divide into 4 patches
        b, n, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        inputlist = [
            x[:, :, :, 0:h_size, 0:w_size],
            x[:, :, :, 0:h_size, (w - w_size):w],
            x[:, :, :, (h - h_size):h, 0:w_size],
            x[:, :, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            outputlist = []
            for i in range(0, 4, nGPUs):
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                with torch.no_grad():
                    model = model.to(self.device)
                    _, _, _, output_batch = model(input_batch.to(self.device))
                outputlist.append(output_batch.data)
        else:
            outputlist = [
                self.chop_forward(patch, model, scale, shave, min_size, nGPUs)
                for patch in inputlist]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        # output = Variable(x.data.new(1, 1, h, w), volatile=True) #UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
        with torch.no_grad():
            output = Variable(x.data.new(1, c, h, w))
        for idx, out in enumerate(outputlist):
            if len(out.shape) < 4:
                outputlist[idx] = out.unsqueeze(0)
        output[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = outputlist[1][:,
                                                        :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] = outputlist[2][:,
                                                        :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = outputlist[3][:, :,
                                                        (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output.float().cpu()


class SOFVSR_RRDB_Model(SOFVSRModel):
    def __init__(self, only_y=False, num_frames=3, num_channels=320, scale=4, SR_net='rrdb', img_ch=3, sr_nf=64, sr_nb=32, sr_gaussian_noise=False, device=None):
        super(SOFVSR_RRDB_Model, self).__init__()
        self.device = device
        
        self.model = SOFVSR(scale=scale, n_frames=num_frames,
                            channels=num_channels, SR_net=SR_net,
                            img_ch=img_ch, sr_nf=sr_nf, sr_nb=sr_nb,
                            sr_gaussian_noise=sr_gaussian_noise).eval()
        self.only_y = only_y
        self.num_frames = num_frames
        self.num_padding = (num_frames - 1) // 2
        self.num_channels = num_channels
        self.scale = scale
        self.SR_net = SR_net
        self.img_ch = img_ch
        self.sr_nf = sr_nf
        self.sr_nb = sr_nb
        self.sr_gaussian_noise = sr_gaussian_noise
