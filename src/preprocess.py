
import torch

def generate_dummy_data(batch_size=64, input_dim=784):
    """
    Generate dummy data for testing the model.
    """
    return torch.randn(batch_size, input_dim)
