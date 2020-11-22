# Video Inference

This repository is an inference repo similar to that of the ESRGAN inference repository, but for various video machine learning models.

## Currently supported architectures

- SOFVSR (@Victorca25's BasicSR Version)

## Additional features

- Automatic scale, number of frames, and number of channels detection
- Automatic beginning and end frame padding so frames 1 and -1 get included in output

## Using this repo

- Place exported video frames in the `input` folder
- Place model in the `models` folder
- `python run.py ./models/video_model.pth`

## Extra flags

- `--input`: Specifies input directory
- `--output`: Specifies output directory
- `--denoise`: Denoises the chroma layer
- `--chop_forward`: Splits tensors to avoid out-of-memory errors

## Planned architecture support

- EDVR
- RRN
- RIFE

## Planned additional features

- Direct video input/output via ffmpeg
