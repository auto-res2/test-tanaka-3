import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from tqdm import tqdm

def evaluate_model(model, test_loader, device, save_dir=None, optimizer_name=None):
    """
    Evaluate a model and optionally save metrics/plots
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
        save_dir: Directory to save plots (optional)
        optimizer_name: Name of optimizer used for training (for labeling)
    
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy (%)
        confusion_mat: Confusion matrix (if classification task)
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if outputs.size(0) != targets.size(0):
                outputs = outputs[:targets.size(0)]
            
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    if len(all_targets) > 0:
        labels = np.unique(all_targets)
        conf_mat = confusion_matrix(all_targets, all_preds, labels=labels)
        
        if save_dir and optimizer_name:
            os.makedirs(save_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                       xticklabels=[str(l) for l in labels], yticklabels=[str(l) for l in labels])
            plt.title(f'Confusion Matrix ({optimizer_name})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f"{save_dir}/{optimizer_name}_confusion_matrix.pdf", format='pdf', bbox_inches='tight')
            plt.close()
    else:
        conf_mat = None
    
    return test_loss, test_acc, conf_mat

def run_synthetic_experiment(optimizers_dict, save_dir=None, num_iters=100, problem_type="quadratic"):
    """
    Run synthetic optimization benchmark
    
    Args:
        optimizers_dict: Dictionary of optimizer classes and parameters
        save_dir: Directory to save plots
        num_iters: Number of iterations to run
        problem_type: Type of problem ('quadratic' or 'rosenbrock')
    
    Returns:
        results: Dictionary of optimization results per optimizer
    """
    print(f"=== Synthetic Experiment: {problem_type.capitalize()} Function Optimization ===")
    torch.manual_seed(0)  # For reproducibility
    
    if problem_type == "quadratic":
        A = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
        b = torch.tensor([1.0, 1.0])
        
        def objective_fn(x):
            return 0.5 * x @ A @ x - b @ x
            
    elif problem_type == "rosenbrock":
        a, b = 1.0, 100.0
        
        def objective_fn(x):
            return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    
    results = {name: [] for name in optimizers_dict.keys()}
    
    for name, (opt_class, opt_params) in optimizers_dict.items():
        print(f"\nRunning optimization with {name}")
        
        x_data = torch.randn(2, requires_grad=True)
        
        if name == "ACM":
            from train import ACMOptimizer
            optimizer = ACMOptimizer([x_data], **opt_params)
        elif name == "Adam":
            optimizer = torch.optim.Adam([x_data], **opt_params)
        elif name == "SGD_mom":
            optimizer = torch.optim.SGD([x_data], **opt_params)
        
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = objective_fn(x_data)
            loss.backward()
            optimizer.step()
            results[name].append(loss.item())
            
            if (i + 1) % (num_iters // 5) == 0 or i == 0:
                print(f"Iter {i+1}/{num_iters} - Loss: {loss.item():.4f}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        for name, losses in results.items():
            plt.plot(losses, label=name)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{problem_type.capitalize()} Function Optimization")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/{problem_type}_optimization.pdf", format='pdf', bbox_inches='tight')
        plt.close()
    
    return results
