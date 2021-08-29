import argparse
import torch
import os
import sys
import progressbar

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
parser.add_argument('--exp', default=1, type=int,
                    help='RIFE exponential interpolation amount')
parser.add_argument('--fp16', default=False, type=bool,
                    help='Use floating-point 16 mode for faster inference')
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

if args.fp16:
    torch.set_default_tensor_type(
        torch.HalfTensor if args.cpu else torch.cuda.HalfTensor
    )


def main():
    state_dict = torch.load(args.model)

    model = get_model_from_state_dict(state_dict, device)

    model.load_state_dict(state_dict)

    # Case for if input and output are video files, read/write with ffmpeg
    if is_video:
        from utils.io_classes.video_io import VideoIO
        io = VideoIO(args.output, model.scale, crf=args.crf, exp=args.exp)
    # Regular case with input/output frame images
    else:
        from utils.io_classes.image_io import ImageIO
        io = ImageIO(args.output)

    # Feed input path to i/o
    io.set_input(args.input)

    # Pad beginning and end frames so they get included in output
    io.pad_data(model.num_padding)

    # Pass i/o into model
    model.set_io(io)

    # Inference loop
    # , redirect_stdout=True):
    for idx in progressbar.progressbar(range(model.num_padding, len(io) - model.num_padding)):

        LR_list = model.get_frames(idx)

        model.inference(LR_list, args)

    # Close output stream (if video)
    model.io.close()


if __name__ == '__main__':
    main()
