import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time

class ACMOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, beta=0.9, curvature_influence=0.1):
        defaults = dict(lr=lr, beta=beta, curvature_influence=curvature_influence)
        super(ACMOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            curvature_influence = group['curvature_influence']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(grad).detach()
                    p.data.add_(-lr * grad)
                else:
                    buf = state['momentum_buffer']
                    curvature_est = (grad - buf).abs()
                    adaptive_lr = lr / (1.0 + curvature_influence * curvature_est)
                    buf.mul_(beta).add_(grad, alpha=1 - beta)
                    p.data.add_(-adaptive_lr * buf)
        
        return loss

def train_model(model, train_loader, optimizer_name, optimizer_params, device, epochs, save_dir=None):
    """
    Train a model with specified optimizer and parameters
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer_name: String name of optimizer (ACM, Adam, SGD_mom)
        optimizer_params: Dictionary containing optimizer parameters
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        save_dir: Directory to save plots (optional)
    
    Returns:
        training_losses: List of training losses per epoch
        training_accs: List of training accuracies per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "ACM":
        optimizer = ACMOptimizer(
            model.parameters(),
            lr=optimizer_params.get('lr', 0.01),
            beta=optimizer_params.get('beta', 0.9),
            curvature_influence=optimizer_params.get('curvature_influence', 0.1)
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_params.get('lr', 0.001)
        )
    elif optimizer_name == "SGD_mom":
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_params.get('lr', 0.01),
            momentum=optimizer_params.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    training_losses = []
    training_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            if outputs.size(0) != targets.size(0):
                outputs = outputs[:targets.size(0)]
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix(
                loss=running_loss/(batch_idx+1),
                acc=100.*correct/total
            )
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        training_losses.append(epoch_loss)
        training_accs.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), training_losses)
        plt.title(f'Training Loss ({optimizer_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{save_dir}/{optimizer_name}_training_loss.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), training_accs)
        plt.title(f'Training Accuracy ({optimizer_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(f"{save_dir}/{optimizer_name}_training_acc.pdf", format='pdf', bbox_inches='tight')
        plt.close()
    
    return training_losses, training_accs
