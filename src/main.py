
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import geoopt
import time

from train import run_experiment1, SelfAdaptiveHyperbolicAutoencoder
from evaluate import run_experiment2
from preprocess import generate_dummy_data

os.makedirs('logs', exist_ok=True)

def run_experiment3(epochs=20, batch_size=64, save_dir='logs'):
    print("Running Experiment 3: Self-Adaptive Hyperbolic Autoencoder")
    input_dim = 28 * 28
    latent_dim = 32
    
    model = SelfAdaptiveHyperbolicAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    losses = []
    curvature_list = []
    for epoch in range(epochs):
        x = generate_dummy_data(batch_size, input_dim)
        optimizer.zero_grad()
        reconstruction, hyperbolic_latent = model(x)
        loss = criterion(reconstruction, x)
        loss.backward()
        optimizer.step()
        var_val = model.update_curvature(hyperbolic_latent)
        losses.append(loss.item())
        curvature_list.append(float(model.adaptive_curvature))
        if epoch % 5 == 0:
            print(f"Epoch {epoch} - Loss {loss.item():.4f} - Adaptive Curvature {float(model.adaptive_curvature):.4f} - Radial Variance {var_val:.4f}")
    
    plt.figure()
    plt.plot(range(epochs), losses, marker='o', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Training Loss: Self-Adaptive Hyperbolic Autoencoder")
    plt.grid(True)
    plt.legend()
    filename_loss = os.path.join(save_dir, "training_loss_adaptive_pair1.pdf")
    plt.savefig(filename_loss, format="pdf")
    print(f"Experiment 3 loss plot saved as: {filename_loss}")
    plt.close()
    
    plt.figure()
    plt.plot(range(epochs), curvature_list, marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Adaptive Curvature")
    plt.title("Adaptive Curvature Evolution over Training Epochs")
    plt.grid(True)
    filename_curv = os.path.join(save_dir, "curvature_adaptive_pair1.pdf")
    plt.savefig(filename_curv, format="pdf")
    print(f"Experiment 3 curvature plot saved as: {filename_curv}")
    plt.close()

def test_all_experiments():
    print("Starting tests for all experiments...")
    run_experiment1(epochs=10, batch_size=32)  # fewer epochs so test finishes rapidly
    run_experiment2(num_samples=64)
    run_experiment3(epochs=10, batch_size=32)
    print("All experiments ran successfully.")

if __name__ == '__main__':
    start_time = time.time()
    test_all_experiments()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    status_enum = "stopped"
    print(f"Status: {status_enum}")
