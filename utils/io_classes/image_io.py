from utils.io_classes.base_io import BaseIO

import numpy as np
import cv2
import os


class ImageIO(BaseIO):
    """
    Image Sequence Input & Output

    Arguments
        output_path (str): The path to write the images to
    """
    def __init__(self, output_path):
        super(ImageIO, self).__init__(output_path)

        self.count = 0

    def set_input(self, input_folder):
        """
        Load file paths with correct image extensions and feed data.

        Arguments:
            input_folder (str): The folder to recursively grab paths from.
        """
        images = []
        for root, _, files in os.walk(input_folder):
            for file in sorted(files):
                if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga']:
                    images.append(os.path.join(root, file))
        self.feed_data(images)

    def save_frames(self, frames):
        """
        Save frame data as images.

        Arguments:
            frames (ndarray, list): The image data to be written
        """
        if not isinstance(frames, list):
            frames = [frames]
        # TODO: Re-add ability to save with original name
        for img in frames:
            cv2.imwrite(os.path.join(self.output_path,
                                     f'{(self.count):08}.png'), img)
            self.count += 1

    def __getitem__(self, idx):
        return cv2.imread(self.data[idx], cv2.IMREAD_COLOR)
