"""
Evaluation module for D3SC experiments.
Contains functions to evaluate models and compute metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2


def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1, img2: Input images
        
    Returns:
        PSNR value
    """
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))


def evaluate_texture_models(model_mini, model_iter, input_sample, target_sample):
    """
    Evaluate texture refinement models.
    
    Args:
        model_mini: Mini-step texture model
        model_iter: Iterative texture model
        input_sample: Input texture sample
        target_sample: Target texture sample
        
    Returns:
        Dictionary with evaluation metrics
    """
    model_mini.eval()
    model_iter.eval()
    with torch.no_grad():
        refined_mini = model_mini(input_sample).squeeze().permute(1, 2, 0).numpy()
        
        iter_output = input_sample
        for _ in range(3):
            iter_output = model_iter(iter_output)
        refined_iter = iter_output.squeeze().permute(1, 2, 0).numpy()
        
        target_img = target_sample.permute(1, 2, 0).numpy()
        
        print(f"Target image shape: {target_img.shape}")
        print(f"Refined mini shape: {refined_mini.shape}")
        print(f"Refined iter shape: {refined_iter.shape}")
        
        ssim_mini = ssim(target_img, refined_mini, channel_axis=2, data_range=1.0, win_size=3)
        ssim_iter = ssim(target_img, refined_iter, channel_axis=2, data_range=1.0, win_size=3)
        psnr_mini = compute_psnr(target_img, refined_mini)
        psnr_iter = compute_psnr(target_img, refined_iter)
    
    return {
        "ssim_mini": ssim_mini,
        "psnr_mini": psnr_mini,
        "ssim_iter": ssim_iter,
        "psnr_iter": psnr_iter
    }


def plot_texture_metrics(metrics, save_path="texture_metrics_comparison.pdf"):
    """
    Plot texture metrics comparison.
    
    Args:
        metrics: Dictionary with metrics
        save_path: Path to save the plot
    """
    labels = ['SSIM', 'PSNR']
    mini_vals = [metrics["ssim_mini"], metrics["psnr_mini"]]
    iter_vals = [metrics["ssim_iter"], metrics["psnr_iter"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, mini_vals, width, label='Mini-step')
    plt.bar(x + width/2, iter_vals, width, label='Iterative')
    plt.xticks(x, labels)
    plt.ylabel('Metric Value')
    plt.title('Texture Refinement: Quality Metrics Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved texture metrics plot as {save_path}")


def setup_render_scene():
    """
    Setup a 3D rendering scene with a sample object.
    
    Returns:
        pyrender Scene object
    """
    import trimesh
    import pyrender
    
    trimesh_mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    return scene


def render_multi_views(scene, camera, poses, renderer):
    """
    Render multiple views of a 3D scene.
    
    Args:
        scene: pyrender Scene object
        camera: pyrender Camera object
        poses: Dictionary of camera poses
        renderer: pyrender OffscreenRenderer
        
    Returns:
        Dictionary of rendered views
    """
    import pyrender
    
    rendered_views = {}
    for view_name, pose in poses.items():
        camera_node = scene.add(camera, pose=pose)
        color, _ = renderer.render(scene)
        rendered_views[view_name] = color
        scene.remove_node(camera_node)
    return rendered_views


def compute_multi_view_consistency(views):
    """
    Compute consistency between multiple views.
    
    Args:
        views: Dictionary of rendered views
        
    Returns:
        Consistency score (higher is better)
    """
    base = views["front"]
    consistency_scores = []
    for key in views:
        if key != "front":
            base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
            other_gray = cv2.cvtColor(views[key], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
            ssim_index = ssim(base_gray, other_gray, data_range=1.0)
            consistency_scores.append(ssim_index)
    return np.mean(consistency_scores)


def plot_multi_views(views, save_path="multi_view_rendering.pdf"):
    """
    Plot multiple rendered views.
    
    Args:
        views: Dictionary of rendered views
        save_path: Path to save the plot
    """
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (view_name, image) in zip(axs, views.items()):
        ax.imshow(image)
        ax.set_title(view_name.capitalize())
        ax.axis('off')
    plt.suptitle("Rendered Views from Multi-View Consistency Experiment")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved multi-view rendering figure as {save_path}")
