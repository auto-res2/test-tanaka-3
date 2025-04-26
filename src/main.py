"""
Main script to run D3SC experiments.
Implements three experiments:
1. Single-Step Geometry Distillation
2. Texture Refinement via Mini Score Distillation
3. Hybrid Forward Pass for Multi-View Consistency
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import cv2

from preprocess import create_geometry_dataloader, create_texture_dataloader, get_texture_data
from train import (GeometryGenerator, IterativeGeometryGenerator, TextureRefinementNet,
                   train_geometry_model, train_texture_net, train_iterative_texture_net)
from evaluate import (compute_psnr, evaluate_texture_models, plot_texture_metrics)

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.experiment_config import (TEST_MODE, GEOMETRY_CONFIG, TEXTURE_CONFIG, 
                                      MULTIVIEW_CONFIG)


def experiment_single_step_geometry(test_mode=False):
    """
    Experiment 1: Single-Step Geometry Distillation Efficiency and Quality
    
    Args:
        test_mode: Whether to run in test mode with reduced data
    
    Returns:
        Dictionary with experiment results
    """
    print("\n--- Running Experiment 1: Single-Step Geometry Distillation ---")
    
    noise_dim = GEOMETRY_CONFIG['noise_dim']
    epochs = 2 if test_mode else GEOMETRY_CONFIG['epochs']
    
    loader = create_geometry_dataloader(GEOMETRY_CONFIG, test_mode)
    
    print("Training single-step geometry generator...")
    model_single = GeometryGenerator(noise_dim=noise_dim)
    optimizer_single = optim.Adam(model_single.parameters(), lr=GEOMETRY_CONFIG['learning_rate'])
    
    t0 = time.time()
    mem_usage_single = memory_usage((train_geometry_model, (model_single, loader, optimizer_single, epochs)))
    t1 = time.time()
    loss_single = train_geometry_model(model_single, loader, optimizer_single, epochs)
    print("Single-step training time: {:.2f} seconds".format(t1-t0))
    print("Peak memory usage (single-step): {:.2f} MB\n".format(max(mem_usage_single)))
    
    print("Training iterative (multi-step) geometry generator...")
    base_gen = GeometryGenerator(noise_dim=noise_dim)
    model_iter = IterativeGeometryGenerator(base_gen, iterations=3)
    optimizer_iter = optim.Adam(model_iter.parameters(), lr=GEOMETRY_CONFIG['learning_rate'])
    
    t0 = time.time()
    mem_usage_iter = memory_usage((train_geometry_model, (model_iter, loader, optimizer_iter, epochs)))
    t1 = time.time()
    loss_iter = train_geometry_model(model_iter, loader, optimizer_iter, epochs)
    print("Iterative training time: {:.2f} seconds".format(t1-t0))
    print("Peak memory usage (iterative): {:.2f} MB\n".format(max(mem_usage_iter)))
    
    plt.figure()
    plt.plot(loss_single, label="Single-Step")
    plt.plot(loss_iter, label="Iterative")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curves: Geometry Generation")
    plt.legend()
    plt.tight_layout()
    pdf_filename = os.path.join("logs", "training_loss_geometry.pdf")
    plt.savefig(pdf_filename)
    print(f"Saved loss plot as {pdf_filename}\n")
    
    return {
        "loss_single": loss_single, 
        "loss_iter": loss_iter,
        "mem_usage_single": max(mem_usage_single),
        "mem_usage_iter": max(mem_usage_iter)
    }


def experiment_texture_refinement(test_mode=False):
    """
    Experiment 2: Texture Refinement via Mini Score Distillation
    
    Args:
        test_mode: Whether to run in test mode with reduced data
    
    Returns:
        Dictionary with experiment results
    """
    print("\n--- Running Experiment 2: Texture Refinement via Mini Score Distillation ---")
    
    epochs = 2 if test_mode else TEXTURE_CONFIG['epochs']
    
    loader = create_texture_dataloader(TEXTURE_CONFIG, test_mode)
    
    print("Training texture refinement network with mini-step updates...")
    model_texture_mini = TextureRefinementNet()
    optimizer_mini = optim.Adam(model_texture_mini.parameters(), lr=TEXTURE_CONFIG['learning_rate'])
    train_texture_net(model_texture_mini, loader, optimizer_mini, epochs)
    
    print("Training texture refinement network with iterative full updates...")
    model_texture_iter = TextureRefinementNet()
    optimizer_iter = optim.Adam(model_texture_iter.parameters(), lr=TEXTURE_CONFIG['learning_rate'])
    train_iterative_texture_net(model_texture_iter, loader, optimizer_iter, epochs, iterations=3)
    
    inputs, targets = get_texture_data(1, TEXTURE_CONFIG['texture_size'])
    
    metrics = evaluate_texture_models(model_texture_mini, model_texture_iter, 
                                    inputs[0:1], targets[0])
    
    print("Mini-step update metrics: SSIM = {:.4f}, PSNR = {:.4f}".format(
        metrics["ssim_mini"], metrics["psnr_mini"]))
    print("Iterative update metrics: SSIM = {:.4f}, PSNR = {:.4f}".format(
        metrics["ssim_iter"], metrics["psnr_iter"]))
    
    pdf_filename = os.path.join("logs", "texture_metrics_comparison.pdf")
    plot_texture_metrics(metrics, pdf_filename)
    
    return metrics


def experiment_multi_view_consistency(test_mode=False):
    """
    Experiment 3: Hybrid Forward Pass for Multi-View Consistency
    
    Args:
        test_mode: Whether to run in test mode with reduced data
    
    Returns:
        Dictionary with experiment results
    """
    print("\n--- Running Experiment 3: Hybrid Forward Pass for Multi-View Consistency ---")
    
    dummy_model = nn.Sequential(nn.Linear(10, 10))
    dummy_input = torch.randn(2, 10)
    standard_loss = nn.functional.mse_loss(dummy_model(dummy_input), dummy_input)
    
    if test_mode:
        print("Running in test mode - skipping 3D rendering")
        # Use dummy values for consistency
        consistency = 0.85  # Dummy consistency value
        hybrid_loss_weight = MULTIVIEW_CONFIG['hybrid_loss_weight']
        hybrid_loss = standard_loss + hybrid_loss_weight * (1 - consistency)
        
        print("Dummy multi-view consistency score: {:.4f}".format(consistency))
        print("Standard loss (without hybrid component): {:.4f}".format(standard_loss.item()))
        print("Hybrid loss (with consistency component): {:.4f}".format(hybrid_loss.item()))
        
        plt.figure(figsize=(9, 3))
        for i, view_name in enumerate(["front", "side", "back"]):
            plt.subplot(1, 3, i+1)
            plt.text(0.5, 0.5, f"{view_name} view\n(dummy in test mode)", 
                    ha='center', va='center')
            plt.axis('off')
        plt.suptitle("Multi-View Consistency (Test Mode - No Rendering)")
        pdf_filename = os.path.join("logs", "multi_view_rendering.pdf")
        plt.savefig(pdf_filename)
        print(f"Saved dummy multi-view figure as {pdf_filename}")
        
        return {
            "consistency": consistency,
            "standard_loss": standard_loss.item(),
            "hybrid_loss": hybrid_loss.item()
        }
    
    import pyrender
    import trimesh
    from evaluate import (setup_render_scene, render_multi_views, 
                         compute_multi_view_consistency, plot_multi_views)
    
    width = MULTIVIEW_CONFIG['renderer_width']
    height = MULTIVIEW_CONFIG['renderer_height']
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    
    camera_poses = {
        "front": np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2.5],
            [0, 0, 0, 1]
        ]),
        "side": np.array([
            [0, 0, 1, 2.5],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        "back": np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 2.5],
            [0, 0, 0, 1]
        ])
    }
    
    scene = setup_render_scene()
    views = render_multi_views(scene, camera, camera_poses, renderer)
    
    consistency = compute_multi_view_consistency(views)
    print("Multi-view consistency score: {:.4f}".format(consistency))
    
    hybrid_loss_weight = MULTIVIEW_CONFIG['hybrid_loss_weight']
    hybrid_loss = standard_loss + hybrid_loss_weight * (1 - consistency)
    
    print("Standard loss (without hybrid component): {:.4f}".format(standard_loss.item()))
    print("Hybrid loss (with consistency component): {:.4f}".format(hybrid_loss.item()))
    
    pdf_filename = os.path.join("logs", "multi_view_rendering.pdf")
    plot_multi_views(views, pdf_filename)
    
    return {
        "consistency": consistency,
        "standard_loss": standard_loss.item(),
        "hybrid_loss": hybrid_loss.item()
    }


def run_smoke_test():
    """
    Run a quick smoke test of all experiments.
    """
    print("----- Starting Smoke Test for All Experiments -----")
    
    geo_results = experiment_single_step_geometry(test_mode=True)
    
    tex_results = experiment_texture_refinement(test_mode=True)
    
    mv_results = experiment_multi_view_consistency(test_mode=True)
    
    print("----- Smoke Test Completed Successfully -----")
    print("\nSummary of Results:")
    print("Experiment 1 - Geometry Losses:", geo_results)
    print("Experiment 2 - Texture Metrics:", tex_results)
    print("Experiment 3 - Multi-View Metrics:", mv_results)
    
    print("\nSetting status_enum to 'stopped'")
    os.environ["status_enum"] = "stopped"


if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)
    
    plt.switch_backend('agg')
    
    if TEST_MODE:
        run_smoke_test()
    else:
        print("Running full D3SC experiments...")
        
        geo_results = experiment_single_step_geometry(test_mode=False)
        
        tex_results = experiment_texture_refinement(test_mode=False)
        
        mv_results = experiment_multi_view_consistency(test_mode=False)
        
        print("\nAll experiments completed successfully.")
        print("\nSummary of Results:")
        print("Experiment 1 - Geometry:", geo_results)
        print("Experiment 2 - Texture:", tex_results)
        print("Experiment 3 - Multi-View:", mv_results)
        
        print("\nSetting status_enum to 'stopped'")
        os.environ["status_enum"] = "stopped"
