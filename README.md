# Video Inference

This repository is an inference repo similar to that of the ESRGAN inference repository, but for various video machine learning models.

## Currently supported architectures

- SOFVSR (@Victorca25's BasicSR Version)

## Additional features

- Automatic scale, number of frames, and number of channels detection
- Automatic beginning and end frame padding so frames 1 and -1 get included in output
- Direct video input and output through ffmpeg

## Using this repo

Requirements: `numpy, opencv-python, pytorch`

Optional requirements: `ffmpeg-python` (to use video input/output)

### Upscaling exported frames

- Place exported video frames in the `input` folder
- Place model in the `models` folder
- Example: `python run.py ./models/video_model.pth`

### Upscaling video files

- Place model in the `models` folder
- Set `--input` to your input video
- Set `--output` to your output video
- Example: `python run.py ./models/video_model.pth --input "./input/input_video.mp4" --output "./output/output_video.mp4"`

## Extra flags

- `--input`: Specifies input directory or file
- `--output`: Specifies output directory or file
- `--denoise`: Denoises the chroma layer
- `--chop_forward`: Splits tensors to avoid out-of-memory errors
- `--crf`: The crf (quality) of the output video when using video input/output. Defaults to 0 (lossless)

## Planned architecture support

- EDVR
- RRN
- RIFE

## Planned additional features

- More FFMPEG options
