"""
Usage:
language_feat_exps/open_vocab_seg_holicity.py --downsample_ratio 0.1

This script evaluates the open-vocab semseg performance on the Holicity dataset using language features.
"""

import os
import numpy as np
import torch
import datetime
import sys
import argparse
import open_clip

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..") # noqa

from plyfile import PlyData
from tqdm import tqdm
from metadata.holicity import Holicity_LABELS
from pathlib import Path
from scipy.spatial import cKDTree as KDTree
from transformers import AutoModel, AutoTokenizer
from lseg import LSegNet
import clip
import sys
import open3d as o3d

from matplotlib.colors import hsv_to_rgb
def generate_distinct_colors(n=100, seed=42):
    np.random.seed(seed)
    
    # Evenly space hues, randomize saturation and value a bit
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)  # shuffle to prevent similar colors being close in order
    saturations = np.random.uniform(0.6, 0.9, n)
    values = np.random.uniform(0.7, 0.95, n)
    
    hsv_colors = np.stack([hues, saturations, values], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors

# Example usage
Holicty_COLORS = generate_distinct_colors(10)



def load_scene_list(val_split_path):
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if line.strip()]
    return scene_ids

def read_ply_file_3dgs(file_path):
    ply_data = PlyData.read(file_path)
    vertex = ply_data["vertex"]
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    opacity = vertex["opacity"]
    xyz = np.stack([x, y, z], axis=-1)
    return xyz, opacity

def clustering_voting(pred, instance_labels, ignore_index):
    """ 
    Args:
        pred (np.ndarray): Predicted semantic labels for each point, shape (N,)
        instance_labels (np.ndarray): Instance ID for each point, shape (N,)
        ignore_index (int): Instance ID value to ignore (e.g., -1 for background)
    Returns:
        np.ndarray: Updated semantic predictions with consistent labels per instance
    """
    # Ensure inputs have the same shape
    if pred.shape != instance_labels.shape:
        print("clustering_voting: prediction and instance arrays must have the same shape, got {} and {}".format(pred.shape, instance_labels.shape))
        return pred
    
    updated_pred = pred.copy()
    unique_instances = np.unique(instance_labels)
    valid_instances = unique_instances[unique_instances != ignore_index]
    
    for instance_id in valid_instances:
        instance_mask = instance_labels == instance_id
        instance_preds = pred[instance_mask]
        unique_classes, counts = np.unique(instance_preds, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        updated_pred[instance_mask] = majority_class
    
    return updated_pred

@torch.no_grad()
def compute_relevancy_scores(lang_feat: torch.Tensor, 
                             text_feat: torch.Tensor,
                             device: torch.device,
                             bench_name: str = ""):
    lang_feat = lang_feat.to(device, non_blocking=True).float()
    text_feat = text_feat.to(device, non_blocking=True).float()
    # if use_siglip_probabilities: # default True

    logits = torch.matmul(lang_feat, text_feat.t())  # (N, C)
    # probs = torch.sigmoid(logits)  # (N, C)
    top1_probs, top1_indices = torch.topk(logits, k=1, dim=1)  # (N, 1)
    mask = top1_probs[:, 0] > 0.
    top1_indices[~mask] = -1 # if lang_feat is all zeros
    # if bench_name == "scannet20":
    #     top1_indices[~mask] = -1 # use "other" class
    # else:
    #     top1_indices[~mask] = -1  # Use -1 as ignore index
    return top1_indices.cpu().numpy()

def save_results_to_file(log_path, results_str, args):
    """Save the results to a text file with relevant parameters in the filename."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Write results to the file
    with open(log_path, 'w') as f:
        # Write command line arguments first
        f.write("Command line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        
        # Write experiment results
        f.write(results_str)
    
    print(f"\nResults saved to: {log_path}")

def main():
    text = ["laptop", "this is a laptop"]
    model_name = "clip"
    # hanger
    # this is a hanger
    # this is a hanger in the corner
    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/checkpoints/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text
    with torch.no_grad():
        text_tokens = clip.tokenize(text)
        text_feat = clip_text_encoder(text_tokens.to('cuda'))  # (C, 512)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1)
        text_feat = text_feat.cpu()
    text_feat = text_feat
    text_feat_npy = text_feat.numpy()
    print("text_feat_npy.shape", text_feat_npy.shape)
    np.save('gradient_backprojection_this_is_a_laptop_text_feat.npy', text_feat_npy)

if __name__ == "__main__":
    main()