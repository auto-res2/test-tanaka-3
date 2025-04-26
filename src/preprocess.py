"""
Preprocessing module for D3SC experiments.
Contains functions to generate synthetic data for experiments.
"""

import numpy as np
import torch
import cv2
from torch.utils.data import TensorDataset, DataLoader


def get_geometry_data(num_samples=1000, noise_dim=128, voxel_size=32):
    """
    Generate synthetic data for geometry experiments.
    
    Args:
        num_samples: Number of samples to generate
        noise_dim: Dimension of noise vectors
        voxel_size: Size of the voxel grid
        
    Returns:
        X: Noise vectors
        Y: Target voxel grids
    """
    X = torch.randn(num_samples, noise_dim)
    Y = (torch.rand(num_samples, voxel_size, voxel_size, voxel_size) > 0.5).float()
    return X, Y


def get_texture_data(num_samples=500, texture_size=64):
    """
    Generate synthetic data for texture refinement experiments.
    
    Args:
        num_samples: Number of samples to generate
        texture_size: Size of texture images
        
    Returns:
        inputs: Random textures
        targets: Blurred versions of inputs as targets
    """
    inputs = np.random.rand(num_samples, 3, texture_size, texture_size).astype(np.float32)
    targets = []
    for img in inputs:
        img_HWC = np.transpose(img, (1, 2, 0))
        blurred = cv2.GaussianBlur(img_HWC, (5, 5), 0)
        targets.append(np.transpose(blurred, (2, 0, 1)))
    targets = np.array(targets, dtype=np.float32)
    return torch.tensor(inputs), torch.tensor(targets)


def create_geometry_dataloader(config, test_mode=False):
    """
    Create dataloader for geometry experiments.
    
    Args:
        config: Configuration dictionary
        test_mode: Whether to use reduced dataset for testing
        
    Returns:
        DataLoader for geometry data
    """
    num_samples = 200 if test_mode else config['num_samples']
    X, Y = get_geometry_data(num_samples, config['noise_dim'], config['voxel_size'])
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)


def create_texture_dataloader(config, test_mode=False):
    """
    Create dataloader for texture refinement experiments.
    
    Args:
        config: Configuration dictionary
        test_mode: Whether to use reduced dataset for testing
        
    Returns:
        DataLoader for texture data
    """
    num_samples = 100 if test_mode else config['num_samples']
    inputs, targets = get_texture_data(num_samples, config['texture_size'])
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
