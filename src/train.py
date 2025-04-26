"""
Training module for D3SC experiments.
Contains models and training functions for the experiments.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from memory_profiler import memory_usage
import matplotlib.pyplot as plt


class GeometryGenerator(nn.Module):
    """
    Generator model for single-step geometry distillation.
    """
    def __init__(self, noise_dim=128, output_dim=32**3):
        super(GeometryGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()  # occupancy probabilities
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size, 32, 32, 32)


class IterativeGeometryGenerator(nn.Module):
    """
    Iterative variant of the geometry generator.
    """
    def __init__(self, base_generator, iterations=3, noise_dim=128):
        super(IterativeGeometryGenerator, self).__init__()
        self.base = base_generator
        self.iterations = iterations
        self.noise_dim = noise_dim
        
        self.projection = nn.Linear(32**3, noise_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.base(x)
        for _ in range(self.iterations - 1):
            flattened = out.view(batch_size, -1)
            projected = self.projection(flattened)
            out = self.base(projected)
        return out


def geometry_loss(pred, target):
    """
    Loss function for geometry generator.
    """
    return nn.functional.mse_loss(pred, target)


def train_geometry_model(model, loader, optimizer, epochs=3):
    """
    Train geometry model.
    
    Args:
        model: Model to train
        loader: DataLoader with training data
        optimizer: Optimizer to use
        epochs: Number of training epochs
        
    Returns:
        losses: List of losses per epoch
    """
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for noise, target in loader:
            optimizer.zero_grad()
            output = model(noise)
            loss = geometry_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(loader)
        losses.append(average_loss)
        print(f"[Geometry] Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")
    return losses


class TextureRefinementNet(nn.Module):
    """
    Network for texture refinement.
    """
    def __init__(self, input_channels=3, output_channels=3):
        super(TextureRefinementNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.conv(x)


def train_texture_net(model, loader, optimizer, epochs=3):
    """
    Train texture refinement network.
    
    Args:
        model: Model to train
        loader: DataLoader with training data
        optimizer: Optimizer to use
        epochs: Number of training epochs
        
    Returns:
        model: Trained model
    """
    model.train()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inp, tgt in loader:
            optimizer.zero_grad()
            output = model(inp)
            loss = loss_fn(output, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        print(f"[Texture] Mini-step Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model


def train_iterative_texture_net(model, loader, optimizer, epochs=3, iterations=3):
    """
    Train texture refinement network with iterative updates.
    
    Args:
        model: Model to train
        loader: DataLoader with training data
        optimizer: Optimizer to use
        epochs: Number of training epochs
        iterations: Number of refinement iterations
        
    Returns:
        model: Trained model
    """
    model.train()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for inp, tgt in loader:
            optimizer.zero_grad()
            output = inp
            for _ in range(iterations):
                output = model(output)
            loss = loss_fn(output, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / batch_count
        print(f"[Texture] Iterative Full-Update Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model
