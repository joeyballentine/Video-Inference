

class BaseIO():
    def __init__(self, output_path):
        self.data = None

        self.output_path = output_path

    def set_input(self):
        pass

    def feed_data(self, data):
        self.data = data

    def pad_data(self, num_padding):
        for _ in range(num_padding):
            self.data.insert(0, self.data[0])
            self.data.append(self.data[-1])

    def save_frames(self):
        pass

    def close(self):
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)