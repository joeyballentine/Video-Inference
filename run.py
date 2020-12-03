import argparse
import torch
import os
import sys
import cv2
import numpy as np

import utils.architectures.SOFVSR_arch as SOFVSR
import utils.architectures.RIFE_arch as RIFE

import utils.common as util
from utils.colors import *
from utils.state_dict_utils import get_model_from_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--input', default='input', help='Input folder')
parser.add_argument('--output', default='output', help='Output folder')
parser.add_argument('--cpu', action='store_true',
                    help='Use CPU instead of CUDA')
parser.add_argument('--denoise', action='store_true',
                    help='Denoise the chroma layers')
parser.add_argument('--chop_forward', action='store_true')
parser.add_argument('--crf', default=0, type=int)
parser.add_argument('--exp', default=2, type=int, help='RIFE exponential interpolation amount')
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

def main():
    state_dict = torch.load(args.model)

    model = get_model_from_state_dict(state_dict, device)

    model.load_state_dict(state_dict)

    # Case for if input and output are video files, read/write with ffmpeg
    # TODO: Refactor this to be less messy
    if is_video:
        # Import ffmpeg here because it is only needed if input/output is video
        import ffmpeg

        # Grabs video metadata information
        probe = ffmpeg.probe(args.input)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        framerate = int(video_stream['r_frame_rate'].split(
            '/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
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
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width * model.scale, height * model.scale))
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
    for _ in range(model.num_padding):
        images.insert(0, images[0])
        images.append(images[-1])

    count = 0
    # Inference loop
    for idx in range(model.num_padding, len(images) - model.num_padding):

        # Only print this if processing frames
        if not is_video:
            img_name = os.path.splitext(os.path.basename(images[idx]))[0]
            print(idx - model.num_padding, img_name)

        model.feed_data(images)

        LR_list = model.get_frames(idx, is_video)

        sr_img = model.inference(LR_list, args)

        # TODO: Refactor this to be less messy
        if not is_video:
            # save images
            if isinstance(sr_img, list):
                for i, img in enumerate(sr_img):
                    # cv2.imwrite(os.path.join(output_folder,
                    #                     f'{os.path.basename(images[idx]).split(".")[0]}_{i}.png'), img)
                    cv2.imwrite(os.path.join(output_folder,
                    f'{(count):08}.png'), img)
                    count += 1
            else:
                cv2.imwrite(os.path.join(output_folder,
                                        os.path.basename(images[idx])), sr_img)
        else:
            # Write SR frame to output video stream
            if isinstance(sr_img, list):
                for img in sr_img:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    process.stdin.write(
                        img
                        .astype(np.uint8)
                        .tobytes()
                    )
            else:
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
