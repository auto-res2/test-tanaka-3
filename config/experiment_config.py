"""
Configuration for the ACM optimizer experiments
"""

general_config = {
    "seed": 42,
    "device": "cuda",  # or "cpu" if no GPU available
    "quick_test": False,  # Set to True for a quick test run
    "save_dir": "logs",
}

synthetic_config = {
    "num_iters": 1000,  # Reduced to 100 for quick_test
    "problems": ["quadratic", "rosenbrock"],
    "optimizers": {
        "ACM": {"lr": 0.1, "beta": 0.9, "curvature_influence": 0.05},
        "Adam": {"lr": 0.1},
        "SGD_mom": {"lr": 0.1, "momentum": 0.9}
    }
}

cifar_config = {
    "model": "cifar_cnn",
    "batch_size": 128,
    "epochs": 20,  # Reduced to 2 for quick_test
    "optimizers": {
        "ACM": {"lr": 0.01, "beta": 0.9, "curvature_influence": 0.1},
        "Adam": {"lr": 0.001},
        "SGD_mom": {"lr": 0.01, "momentum": 0.9}
    }
}

mnist_config = {
    "model": "mnist_cnn",
    "batch_size": 128,
    "epochs": 10,  # Reduced to 2 for quick_test
    "subset_sizes": [10000, 20000, 50000],  # Different training set sizes for ablation
    "optimizers": {
        "ACM": {"lr": 0.01, "beta": 0.9, "curvature_influence": 0.1},
        "Adam": {"lr": 0.001},
        "SGD_mom": {"lr": 0.01, "momentum": 0.9}
    }
}
