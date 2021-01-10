import torch

class BaseModel():
    """
    This is the base models that all model classes should inherit from.
    """
    def __init__(self, device=None):
        self.model = None
        self.device = device if device else (torch.device('cpu' if torch.cuda.is_available() else 'cuda'))

    def load_state_dict(self, state_dict):
        """
        Load the PyTorch state dict pickle file
        """
        self.model.load_state_dict(state_dict)
    
    def set_io(self, io):
        """
        Set the I/O class for the model to use
        """
        self.io = io

    def get_frames(self):
        """
        Get required frames from the I/O class
        """
        pass

    def inference(self):
        """
        Performs the inference step of the model
        """
        pass