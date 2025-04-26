
import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('logs', exist_ok=True)

class HyperbolicEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(HyperbolicEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        self.curvature = 1.0

    def forward(self, x):
        euclidean_latent = self.fc(x)
        manifold = geoopt.PoincareBall(c=self.curvature)
        hyperbolic_latent = manifold.expmap0(euclidean_latent)
        return hyperbolic_latent

def hyperbolic_reparameterize(mu, logvar, manifold, num_samples=1):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    tangent_sample = mu + eps * std  
    return manifold.expmap0(tangent_sample)

def reverse_kl_loss(predicted, target):
    return torch.mean((predicted - target) ** 2)

class HyperbolicAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(HyperbolicAutoencoder, self).__init__()
        self.encoder = HyperbolicEncoder(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        hyperbolic_latent = self.encoder(x)
        manifold = geoopt.PoincareBall(c=self.encoder.curvature)
        tangent_latent = manifold.logmap0(hyperbolic_latent)
        reconstruction = self.decoder(tangent_latent)
        return reconstruction, hyperbolic_latent

class SelfAdaptiveHyperbolicAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SelfAdaptiveHyperbolicAutoencoder, self).__init__()
        self.encoder = HyperbolicEncoder(input_dim, latent_dim)
        self.adaptive_curvature = nn.Parameter(torch.tensor(1.0))
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        c_val = float(self.adaptive_curvature)
        manifold = geoopt.PoincareBall(c=c_val)
        euclidean_latent = self.encoder.fc(x)
        hyperbolic_latent = manifold.expmap0(euclidean_latent)
        tangent_latent = manifold.logmap0(hyperbolic_latent)
        reconstruction = self.decoder(tangent_latent)
        return reconstruction, hyperbolic_latent

    def update_curvature(self, latent):
        """
        Update the curvature based on the variance of the radial distances measured in tangent space.
        """
        c_val = float(self.adaptive_curvature)
        manifold = geoopt.PoincareBall(c=c_val)
        tangent = manifold.logmap0(latent)
        radial = torch.norm(tangent, dim=-1)
        variance = torch.var(radial)
        with torch.no_grad():
            if variance > 1.0:
                self.adaptive_curvature.data *= 0.95
            else:
                self.adaptive_curvature.data *= 1.05
        return float(variance)

def run_experiment1(epochs=20, batch_size=64, save_dir='logs'):
    print("Running Experiment 1: Hyperbolic Reparameterization via Reverse KL Loss")
    input_dim = 28 * 28   # e.g. flattened MNIST images
    latent_dim = 32

    model = HyperbolicAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = reverse_kl_loss

    losses = []
    for epoch in range(epochs):
        x = torch.randn(batch_size, input_dim)  # dummy data
        optimizer.zero_grad()
        reconstruction, _ = model(x)
        loss = criterion(reconstruction, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch} - Loss {loss.item():.4f}")
    
    plt.figure()
    plt.plot(range(epochs), losses, marker='o')
    plt.title("Training Loss: Hyperbolic Reparameterization")
    plt.xlabel("Epoch")
    plt.ylabel("Reverse KL Loss (MSE approximation)")
    plt.grid(True)
    filename = os.path.join(save_dir, "training_loss_hyperbolic_pair1.pdf")
    plt.savefig(filename, format="pdf")
    print(f"Experiment 1 plot saved as: {filename}")
    plt.close()
    
    return model
