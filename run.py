import argparse
import torch
import os
import sys
import cv2
import numpy as np

import utils.architectures.SOFVSR_arch as SOFVSR
from torch.autograd import Variable
import utils.common as util
from utils.colors import *

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--input', default='input', help='Input folder')
parser.add_argument('--output', default='output', help='Output folder')
parser.add_argument('--cpu', action='store_true',
                    help='Use CPU instead of CUDA')
parser.add_argument('--denoise', action='store_true',
                    help='Denoise the chroma layers')
parser.add_argument('--chop_forward', action='store_true')
parser.add_argument('--crf', default=0)
args = parser.parse_args()

is_video = False
if not os.path.exists(args.input):
    print('Error: Folder [{:s}] does not exist.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.input) and args.input.split('.')[-1].lower() in ['mp4', 'mkv', 'm4v', 'gif']:
    is_video = True
    if args.output.split('.')[-1].lower() not in ['mp4', 'mkv', 'm4v', 'gif']:
        print('Error: Output [{:s}] is not a file.'.format(args.input))
        sys.exit(1)
elif not os.path.isfile(args.input) and not os.path.isfile(args.output) and not os.path.exists(args.output):
    os.mkdir(args.output)

device = torch.device('cpu' if args.cpu else 'cuda')

input_folder = os.path.normpath(args.input)
output_folder = os.path.normpath(args.output)

def chop_forward(x, model, scale, shave=16, min_size=5000, nGPUs=1):
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
                model = model.to(device)
                _, _, _, output_batch = model(input_batch.to(device))
            outputlist.append(output_batch.data)
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    # output = Variable(x.data.new(1, 1, h, w), volatile=True) #UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
    with torch.no_grad():
        output = Variable(x.data.new(1, 1, h, w))
    for idx, out in enumerate(outputlist):
        if len(out.shape) < 4:
            outputlist[idx] = out.unsqueeze(0)
    output[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output.float().cpu()

def main():
    state_dict = torch.load(args.model)

    # Automatic scale detection
    keys = state_dict.keys()
    if 'OFR.SR.3.weight' in keys:
        scale = 1
    elif 'SR.body.6.bias' in keys:
        # 2 and 3 share the same architecture keys so here we check the shape
        if state_dict['SR.body.3.weight'].shape[0] == 256:
            scale = 2
        elif state_dict['SR.body.3.weight'].shape[0] == 576:
            scale = 3
    elif 'SR.body.9.bias' in keys:
        scale = 4
    else:
        raise ValueError('Scale could not be determined from model')

    # Extract num_frames from model
    frame_size = state_dict['SR.body.0.weight'].shape[1]
    num_frames = ((frame_size - 1) // scale ** 2) + 1

    # Extract num_channels
    num_channels = state_dict['OFR.RNN1.0.weight'].shape[0]
    
    # Create model
    model = SOFVSR.SOFVSR(scale=scale, n_frames=num_frames, channels=num_channels)
    model.load_state_dict(state_dict)

    # Case for if input and output are video files, read/write with ffmpeg
    if is_video:
        # Import ffmpeg here because it is only needed if input/output is video
        import ffmpeg

        # Grabs video metadata information
        probe = ffmpeg.probe(args.input)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        framerate = int(video_stream['r_frame_rate'].split('/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
        vcodec = 'libx264'
        crf = args.crf

        # Imports video to buffer
        out, _ = (
            ffmpeg
            .input(args.input)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
        )
        # Reads video buffer into numpy array
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )

        # Convert numpy array into frame list 
        images = []
        for i in range(video.shape[0]):
            frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
            images.append(frame)

        # Open output file writer
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width * scale, height * scale))
                .output(args.output, pix_fmt='yuv420p', vcodec=vcodec, r=framerate, crf=crf, preset='veryfast')
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    # Regular case with input/output frame images
    else:
        images = []
        for root, _, files in os.walk(input_folder):
            for file in sorted(files):
                if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga']:
                    images.append(os.path.join(root, file))
    
    # Pad beginning and end frames so they get included in output
    images.insert(0, images[0])
    images.append(images[-1])

    # Inference loop
    for idx, path in enumerate(images[1:-1], 0):
        idx_center = (num_frames - 1) // 2 
        idx_frame = idx
        
        # Only print this if processing frames
        if not is_video:
            img_name = os.path.splitext(os.path.basename(path))[0]
            print(idx_frame, img_name)  

        # read LR frames
        LR_list = []
        LR_bicubic = None
        for i_frame in range(num_frames):
            # Last and second to last frames
            if idx == len(images)-2 and num_frames == 3:
                # print("second to last frame:", i_frame)
                if i_frame == 0:
                    LR_img = images[idx] if is_video else cv2.imread(images[idx_frame], cv2.IMREAD_COLOR)
                else:
                    LR_img = images[idx+1] if is_video else cv2.imread(images[idx_frame+1], cv2.IMREAD_COLOR)
            elif idx == len(images)-1 and num_frames == 3:
                # print("last frame:", i_frame)
                LR_img = images[idx] if is_video else cv2.imread(images[idx_frame], cv2.IMREAD_COLOR)
            # Every other internal frame
            else:
                # print("normal frame:", idx_frame)
                LR_img = images[idx+i_frame] if is_video else cv2.imread(images[idx_frame+i_frame], cv2.IMREAD_COLOR)

            # get the bicubic upscale of the center frame to concatenate for SR
            if i_frame == idx_center:
                if args.denoise:
                    LR_bicubic = cv2.blur(LR_img, (3,3))
                else:
                    LR_bicubic = LR_img
                LR_bicubic = util.imresize_np(img=LR_bicubic, scale=scale) # bicubic upscale
            
            # extract Y channel from frames
            # normal path, only Y for both
            LR_img = util.bgr2ycbcr(LR_img, only_y=True)

            # expand Y images to add the channel dimension
            # normal path, only Y for both
            LR_img = util.fix_img_channels(LR_img, 1)

            LR_list.append(LR_img) # h, w, c

        LR = np.concatenate((LR_list), axis=2) # h, w, t

        LR = util.np2tensor(LR, bgr2rgb=False, add_batch=True) # Tensor, [CT',H',W'] or [T, H, W]

        # generate Cr, Cb channels using bicubic interpolation
        LR_bicubic = util.bgr2ycbcr(LR_bicubic, only_y=False)
        LR_bicubic = util.np2tensor(LR_bicubic, bgr2rgb=False, add_batch=True)

        if len(LR.size()) == 4:
            b, n_frames, h_lr, w_lr = LR.size()
            LR = LR.view(b, -1, 1, h_lr, w_lr) # b, t, c, h, w

        if args.chop_forward:

            # crop borders to ensure each patch can be divisible by 2
            _, _, _, h, w = LR.size()
            h = int(h//16) * 16
            w = int(w//16) * 16
            LR = LR[:, :, :, :h, :w]
            if isinstance(LR_bicubic, torch.Tensor):
                SR_cb = LR_bicubic[:, 1, :h * scale, :w * scale]
                SR_cr = LR_bicubic[:, 2, :h * scale, :w * scale]
                            
            SR_y = chop_forward(LR, model, scale).squeeze(0)
            sr_img = ycbcr_to_rgb(torch.stack((SR_y, SR_cb, SR_cr), -3))
        else:

            with torch.no_grad():
                model.to(device)
                _, _, _, fake_H = model(LR.to(device))

            SR = fake_H.detach()[0].float().cpu()
            SR_cb = LR_bicubic[:, 1, :, :]
            SR_cr = LR_bicubic[:, 2, :, :]
        
            sr_img = ycbcr_to_rgb(torch.stack((SR, SR_cb, SR_cr), -3))
        
        sr_img = util.tensor2np(sr_img)  # uint8
        
        if not is_video:
            # save images
            cv2.imwrite(os.path.join(output_folder, os.path.basename(path)), sr_img)
        else:
            # Write SR frame to output video stream
            sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
            process.stdin.write(
                sr_img
                    .astype(np.uint8)
                    .tobytes()
            )

    # Close output stream
    if is_video:
        process.stdin.close()
        process.wait()


if __name__ == '__main__':
    main()
