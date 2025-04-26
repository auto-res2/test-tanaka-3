
import torch
import numpy as np
import matplotlib.pyplot as plt
import geoopt
import os

def geometric_noise_schedule(t, sigma0=1.0, alpha=0.1, beta=0.1):
    """
    Compute noise scales for radial and angular components at time step t.
    t: scalar (or numpy array) representing the time step in [0,1]
    """
    radial_noise = sigma0 * np.exp(-alpha * t)
    angular_noise = sigma0 * (1 - np.exp(-beta * t))
    return radial_noise, angular_noise

def apply_geometric_noise(latent, t, manifold):
    """
    Apply geometric noise to a latent representation in hyperbolic space.
    latent: input latent tensor in hyperbolic space.
    manifold: hyperbolic manifold, e.g., PoincareBall.
    """
    tangent_latent = manifold.logmap0(latent)
    radial = torch.norm(tangent_latent, dim=-1, keepdim=True)
    direction = tangent_latent / (radial + 1e-6)
    
    t_val = t if isinstance(t, float) else t.item()
    radial_noise, angular_noise = geometric_noise_schedule(t_val)
    
    noisy_radial = radial + radial_noise * torch.randn_like(radial)
    noisy_direction = direction + angular_noise * torch.randn_like(direction)
    noisy_direction = noisy_direction / (noisy_direction.norm(dim=-1, keepdim=True) + 1e-6)
    
    noisy_tangent = noisy_radial * noisy_direction
    noisy_latent = manifold.expmap0(noisy_tangent)
    return noisy_latent

def run_experiment2(num_samples=128, save_dir='logs'):
    print("Running Experiment 2: Geometric Noise Scheduling")
    latent_dim = 32
    c_value = 1.0
    manifold = geoopt.PoincareBall(c=c_value)
    
    euclidean_latent = torch.randn(num_samples, latent_dim)
    latent = manifold.expmap0(euclidean_latent)  # map to hyperbolic space
    
    t = 0.5
    noisy_latent = apply_geometric_noise(latent, t, manifold)
    
    tangent_latent = manifold.logmap0(latent)
    tangent_noisy = manifold.logmap0(noisy_latent)
    distances_before = torch.norm(tangent_latent, dim=-1).detach().numpy()
    distances_after = torch.norm(tangent_noisy, dim=-1).detach().numpy()
    
    plt.figure()
    plt.hist(distances_before, bins=20, alpha=0.5, label="Before noise")
    plt.hist(distances_after, bins=20, alpha=0.5, label="After noise")
    plt.xlabel("Radial distance (tangent norm)")
    plt.ylabel("Frequency")
    plt.title(f"Hyperbolic distances: Geometric Noise Scheduling (t = {t:.2f})")
    plt.legend()
    filename = os.path.join(save_dir, "distance_histogram_geometric_pair1.pdf")
    plt.savefig(filename, format="pdf")
    print(f"Experiment 2 plot saved as: {filename}")
    plt.close()
