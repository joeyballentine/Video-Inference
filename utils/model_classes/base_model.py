import torch

class BaseModel():
    def __init__(self, device=None):
        self.model = None
        self.device = device if device else (torch.device('cpu' if torch.cuda.is_available() else 'cuda'))

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def set_io(self, io):
        self.io = io

    def get_frames(self):
        pass

    def inference(self):
        pass