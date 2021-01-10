from utils.io_classes.base_io import BaseIO

import numpy as np
import cv2
import ffmpeg


class VideoIO(BaseIO):
    """
    Video Input & Output

    Arguments
        output_path (str): The path to write the video to
        scale (int): The scale of the output (for VSR)
        exp (int): The exponential interpolation factor (for RIFE)
    """
    def __init__(self, output_path, scale, crf=0, exp=1):
        super(VideoIO, self).__init__(output_path)

        self.crf = crf
        self.scale = scale
        self.exp = exp

        self.process = None

    def set_input(self, input_video):
        """
        Load video file into numpy array using ffmpeg, feed data, and open output stream.

        Arguments:
            input_video (str): The path of the video file to read.
        """
        # Grabs video metadata information
        probe = ffmpeg.probe(input_video)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        framerate = int(video_stream['r_frame_rate'].split(
            '/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
        vcodec = 'libx264'

        # Imports video to buffer
        out, _ = (
            ffmpeg
            .input(input_video)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args('-loglevel', 'error')
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

        self.feed_data(images)

        # Open output file writer
        self.process = (
            ffmpeg
            .input('pipe:', r=framerate*(self.exp**2), format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width * self.scale, height * self.scale))
            .output(self.output_path, pix_fmt='yuv420p', vcodec=vcodec, r=framerate*(self.exp**2), crf=self.crf, preset='veryfast')
            .global_args('-loglevel', 'error')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def save_frames(self, frames):
        """
        Write image data to video output stream.

        Arguments:
            frames (ndarray, list): The image data to be written
        """
        if not isinstance(frames, list):
            frames = [frames]
        for img in frames:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.process.stdin.write(
                img
                .astype(np.uint8)
                .tobytes()
            )

    def close(self):
        """
        Close output stream.
        """
        self.process.stdin.close()
        self.process.wait()

    def __getitem__(self, idx):
        return self.data[idx]