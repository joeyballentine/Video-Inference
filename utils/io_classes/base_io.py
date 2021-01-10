

class BaseIO():
    """
    This is the base Input & Output (I/O) class that all I/O classes should inherit from
    """
    def __init__(self, output_path):
        self.data = None

        self.output_path = output_path

    def set_input(self):
        """
        Initialize everything related to input.
        """
        pass

    def feed_data(self, data):
        """
        Fill the class's data list.
        """
        self.data = data

    def pad_data(self, num_padding):
        """
        Pad data by prepending and appending frames.
        """
        for _ in range(num_padding):
            self.data.insert(0, self.data[0])
            self.data.append(self.data[-1])

    def save_frames(self):
        """
        Save frames to output source.
        """
        pass

    def close(self):
        """
        Close output stream, if applicable.
        """
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)