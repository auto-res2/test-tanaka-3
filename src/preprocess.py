import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_cifar10_data(batch_size=128, data_dir='./data'):
    """
    Prepare CIFAR-10 datasets and dataloaders
    
    Args:
        batch_size: Batch size for DataLoader
        data_dir: Directory to save/load datasets
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        classes: List of class names
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

def get_mnist_data(batch_size=128, data_dir='./data', subset_size=None):
    """
    Prepare MNIST datasets and dataloaders
    
    Args:
        batch_size: Batch size for DataLoader
        data_dir: Directory to save/load datasets
        subset_size: If specified, use only this many samples from training set
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    if subset_size and subset_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)
    
    actual_batch_size = min(batch_size, len(train_dataset) if hasattr(train_dataset, '__len__') else batch_size)
    
    train_loader = DataLoader(
        train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=actual_batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader
