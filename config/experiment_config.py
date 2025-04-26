"""
Configuration file for D3SC experiments.
Contains parameters for all three experiments.
"""

TEST_MODE = True  # Set to True for quick test runs

GEOMETRY_CONFIG = {
    'noise_dim': 128,
    'num_samples': 2000,
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'voxel_size': 32,  # Size of voxel grid (32x32x32)
}

TEXTURE_CONFIG = {
    'num_samples': 200,
    'epochs': 5,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'texture_size': 64,  # Size of texture images (64x64)
}

MULTIVIEW_CONFIG = {
    'hybrid_loss_weight': 0.5,
    'renderer_width': 256,
    'renderer_height': 256,
}
