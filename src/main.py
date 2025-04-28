import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import json
from pathlib import Path

from train import train_model, ACMOptimizer
from evaluate import evaluate_model, run_synthetic_experiment
from preprocess import get_cifar10_data, get_mnist_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.simple_models import create_model
from config.experiment_config import general_config, synthetic_config, cifar_config, mnist_config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def quick_test():
    """
    Run a quick test with minimal iterations to verify code execution
    """
    print("Running quick test mode with minimal iterations...")
    
    general_config["quick_test"] = True
    synthetic_config["num_iters"] = 100
    cifar_config["epochs"] = 1
    mnist_config["epochs"] = 1
    mnist_config["subset_sizes"] = [1000]
    
    run_all_experiments()

def run_all_experiments():
    """
    Run all experiments: synthetic benchmarks, CIFAR-10, and MNIST ablation
    """
    save_dir = general_config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    set_seed(general_config["seed"])
    
    device = torch.device(general_config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: SYNTHETIC OPTIMIZATION BENCHMARK")
    print("="*80)
    
    quick_test_str = " (QUICK TEST)" if general_config["quick_test"] else ""
    for problem in synthetic_config["problems"]:
        optimizers_dict = {
            name: (ACMOptimizer if name == "ACM" else 
                  torch.optim.Adam if name == "Adam" else
                  lambda params, **kwargs: torch.optim.SGD(params, **kwargs),
                  params)
            for name, params in synthetic_config["optimizers"].items()
        }
        
        print(f"\nRunning {problem} benchmark{quick_test_str}...")
        run_synthetic_experiment(
            optimizers_dict,
            save_dir=save_dir,
            num_iters=synthetic_config["num_iters"],
            problem_type=problem
        )
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: CIFAR-10 CNN TRAINING")
    print("="*80)
    
    print("\nPreparing CIFAR-10 data...")
    train_loader, test_loader, classes = get_cifar10_data(
        batch_size=cifar_config["batch_size"]
    )
    
    cifar_results = {}
    for opt_name, opt_params in cifar_config["optimizers"].items():
        print(f"\nTraining CIFAR-10 with {opt_name}{quick_test_str}...")
        
        model = create_model(cifar_config["model"], init_weights=True)
        print(f"Model: {cifar_config['model']}")
        
        train_losses, train_accs = train_model(
            model=model,
            train_loader=train_loader,
            optimizer_name=opt_name,
            optimizer_params=opt_params,
            device=device,
            epochs=cifar_config["epochs"],
            save_dir=save_dir
        )
        
        test_loss, test_acc, conf_mat = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir=save_dir,
            optimizer_name=opt_name
        )
        
        cifar_results[opt_name] = {
            "train_loss": train_losses[-1],
            "train_acc": train_accs[-1],
            "test_loss": test_loss,
            "test_acc": test_acc
        }
    
    print("\nCIFAR-10 Results Summary:")
    for opt_name, metrics in cifar_results.items():
        print(f"{opt_name}: Test Acc = {metrics['test_acc']:.2f}%, Test Loss = {metrics['test_loss']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(list(cifar_results.keys()), [r["test_acc"] for r in cifar_results.values()])
    plt.title("CIFAR-10 Test Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y')
    plt.savefig(f"{save_dir}/cifar10_accuracy_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: MNIST ABLATION STUDY")
    print("="*80)
    
    mnist_results = {size: {} for size in mnist_config["subset_sizes"]}
    
    for subset_size in mnist_config["subset_sizes"]:
        print(f"\nMNIST Ablation with {subset_size} training samples{quick_test_str}...")
        
        mnist_batch_size = min(mnist_config["batch_size"], subset_size // 10) if subset_size < 5000 else mnist_config["batch_size"]
        train_loader, test_loader = get_mnist_data(
            batch_size=mnist_batch_size,
            subset_size=subset_size
        )
        
        for opt_name, opt_params in mnist_config["optimizers"].items():
            print(f"\nTraining MNIST ({subset_size} samples) with {opt_name}...")
            
            model = create_model(mnist_config["model"], init_weights=True)
            
            train_losses, train_accs = train_model(
                model=model,
                train_loader=train_loader,
                optimizer_name=opt_name,
                optimizer_params=opt_params,
                device=device,
                epochs=mnist_config["epochs"],
                save_dir=f"{save_dir}/mnist_{subset_size}"
            )
            
            test_loss, test_acc, conf_mat = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device,
                save_dir=f"{save_dir}/mnist_{subset_size}",
                optimizer_name=opt_name
            )
            
            mnist_results[subset_size][opt_name] = {
                "train_loss": train_losses[-1],
                "train_acc": train_accs[-1],
                "test_loss": test_loss,
                "test_acc": test_acc
            }
    
    plt.figure(figsize=(12, 8))
    for opt_name in mnist_config["optimizers"].keys():
        sizes = mnist_config["subset_sizes"]
        accs = [mnist_results[size][opt_name]["test_acc"] for size in sizes]
        plt.plot(sizes, accs, marker='o', label=opt_name)
    
    plt.title("MNIST Ablation Study: Effect of Training Set Size")
    plt.xlabel("Training Set Size")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/mnist_ablation_study.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    status = {"status_enum": "stopped"}
    with open(f"{save_dir}/status.json", "w") as f:
        json.dump(status, f)
    
    print("\nAll experiments completed successfully!")
    print(f"Results saved to {save_dir}/")
    print("status_enum set to 'stopped'")

if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        run_all_experiments()
