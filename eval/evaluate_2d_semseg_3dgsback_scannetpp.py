"""
Zero-shot 2D semantic segmentation evaluation script for ScanNet and ScanNet++
Evaluates rendered 2D feature maps against ground truth labels using CLIP/SigLIP text embeddings
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from lseg import LSegNet
import argparse
from collections import defaultdict
import logging  
import datetime 
import sys


def load_scene_list(split_path):
    """Load scene IDs from split file."""
    with open(split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if line.strip()]
    return scene_ids

def load_class_names(label_file_path):
    """Load class names from label file."""
    with open(label_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

@torch.no_grad()
def compute_pixel_predictions(feature_map, text_features, device):
    """
    Compute per-pixel predictions using dot product similarity.
    
    Args:
        feature_map: [C, H, W] torch tensor
        text_features: [num_classes, C] torch tensor
        device: torch device
    
    Returns:
        predictions: [H, W] numpy array of predicted class indices
    """
    C, H, W = feature_map.shape
    num_classes = text_features.shape[0]
    
    # Move to device
    feature_map = feature_map.to(device, non_blocking=True)
    text_features = text_features.to(device, non_blocking=True)
    
    # Normalize feature map
    feature_map = feature_map / (feature_map.norm(dim=0, keepdim=True) + 1e-10)
    
    # Reshape feature map for matrix multiplication
    features_flat = feature_map.reshape(C, H * W).t()  # [H*W, C]
    
    # Compute similarity scores
    logits = torch.matmul(features_flat, text_features.t())  # [H*W, num_classes]
    
    # Get predictions
    predictions = torch.argmax(logits, dim=1)  # [H*W]
    predictions = predictions.reshape(H, W).cpu().numpy()
    
    return predictions

def prepare_text_features(text_prompts, model, tokenizer, device, model_type):
    """Prepare text features using CLIP/SigLIP model."""
    if model_type == "clip":
        text_tokens = tokenizer(text_prompts).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    elif model_type == "siglip2":
        inputs = tokenizer(text_prompts, padding="max_length",
                           max_length=64, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_feat = model.get_text_features(**inputs)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    elif model_type == "lseg":    
        with torch.no_grad():
            text = tokenizer(text_prompts)
            text = text.cuda() # text = text.to(x.device) # TODO: need use correct device
            text_feat = model(text) # torch.Size([150, 512])
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu()  # shape: (150, 512)
        text_feat = text_feat 

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return text_feat

def update_confusion_matrix_fast(confusion_mat, gt_labels, predictions, num_classes):
    """Efficiently update confusion matrix."""
    # Flatten arrays
    gt_flat = gt_labels.flatten()
    pred_flat = predictions.flatten()
    
    # Filter out ignore labels (-1)
    valid_mask = gt_flat != -1
    gt_valid = gt_flat[valid_mask]
    pred_valid = pred_flat[valid_mask]
    
    # Update confusion matrix
    for gt, pred in zip(gt_valid, pred_valid):
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            confusion_mat[gt, pred] += 1

def compute_metrics(confusion_mat, ignore_classes=None):
    """Compute IoU and accuracy metrics from confusion matrix."""
    num_classes = confusion_mat.shape[0]
    
    # Arrays to store IoU and Acc for each class
    iou_array = np.full(num_classes, np.nan, dtype=float)
    acc_array = np.full(num_classes, np.nan, dtype=float)
    
    # Compute per-class metrics
    for c in range(num_classes):
        tp = confusion_mat[c, c]
        fp = np.sum(confusion_mat[:, c]) - tp
        fn = np.sum(confusion_mat[c, :]) - tp
        
        # Skip if no points of this class exist
        if (tp + fn) == 0:
            continue
        
        acc = tp / (tp + fn)  # Class accuracy
        denom = (tp + fp + fn)
        iou = (tp / denom) if denom > 0 else 0.0
        
        iou_array[c] = iou
        acc_array[c] = acc
    
    # Mean metrics over valid classes
    valid_mask = ~np.isnan(iou_array)
    valid_acc_vals = acc_array[valid_mask]
    valid_iou_vals = iou_array[valid_mask]
    macc = np.mean(valid_acc_vals) if len(valid_acc_vals) > 0 else 0
    miou = np.mean(valid_iou_vals) if len(valid_iou_vals) > 0 else 0
    
    # Metrics excluding ignore classes if specified
    fg_macc, fg_miou = None, None
    if ignore_classes:
        fg_mask = valid_mask.copy()
        for ignore_class in ignore_classes:
            if ignore_class < num_classes:
                fg_mask[ignore_class] = False
        
        fg_acc_vals = acc_array[fg_mask]
        fg_iou_vals = iou_array[fg_mask]
        fg_macc = np.mean(fg_acc_vals) if len(fg_acc_vals) > 0 else 0
        fg_miou = np.mean(fg_iou_vals) if len(fg_iou_vals) > 0 else 0
    
    return {
        'miou': miou,
        'macc': macc,
        'fg_miou': fg_miou,
        'fg_macc': fg_macc,
        'iou_array': iou_array,
        'acc_array': acc_array
    }

def evaluate_dataset(dataset_name, config, model, tokenizer, device, model_type):
    """Evaluate a single dataset configuration."""
    # Modified: Replaced print with logging.info
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating {dataset_name}")
    logging.info(f"{'='*60}")
    
    # Load scene list
    scene_ids = load_scene_list(config['split_path'])
    logging.info(f"Found {len(scene_ids)} scenes")
    # scene_ids = scene_ids[:1]
    # scene_ids=['0d2ee665be']
    
    # Load class names
    class_names = load_class_names(config['label_file'])
    num_classes = len(class_names)
    logging.info(f"Number of classes: {num_classes}")
    
    # Prepare text features
    text_prompts = [f"this is a {name}" for name in class_names]
    text_features = prepare_text_features(text_prompts, model, tokenizer, device, model_type)
    logging.info(f"Text features shape: {text_features.shape}")
    
    # Initialize confusion matrix
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_pixels = 0
    processed_frames = 0
    
    # Process each scene
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):

        # Get feature and label directories
        feature_dir = Path(config['feature_base']) / scene_id  / "3dgsback_render_feats" 
        label_dir = Path(config['label_base']) / scene_id / config['label_subfolder']
        
        if not feature_dir.exists():
            # Modified: Replaced print with logging.warning
            logging.warning(f"Warning: Feature directory not found for scene {scene_id} in {feature_dir}")
            continue
        
        if not label_dir.exists():
            # Modified: Replaced print with logging.warning
            logging.warning(f"Warning: Label directory not found for scene {scene_id} in {label_dir}")
            continue
        
        # Get all feature files
        feature_files = sorted(feature_dir.glob("*.pth"))
        
        for feat_file in tqdm(feature_files, desc="Processing views..."):
            frame_id = feat_file.stem
            frame_id = frame_id.split('_')[0]  # Extract frame ID
            
            # Construct label file path
            if dataset_name.startswith("scannetpp"):
                label_file = label_dir / f"{frame_id}.JPG.npy"
            else:
                label_file = label_dir / f"{frame_id}.npy"
            
            if not label_file.exists():
                # Modified: Replaced print with logging.warning
                logging.warning(f"Warning: Label file not found for frame {frame_id} in scene {scene_id}")
                continue
            
            # Load feature map and labels
            try:
                feature_map =  torch.load(feat_file, map_location='cpu', weights_only=True)  # [C, H, W]
                feature_map = feature_map.permute(2, 0, 1)  # Convert to [H, W, C] format
                feature_map = feature_map.float()  # Ensure it's a flat tensor
                text_features = text_features.float()  # Ensure text features are float
                # print("feature_map shape:", feature_map.shape)

                gt_labels = np.load(label_file)  # [H, W]
                
                # Check dimensions match
                if feature_map.shape[1:] != gt_labels.shape:
                    # print("feature_map shape:", feature_map.shape)
                    # print("gt_labels shape:", gt_labels.shape)
                    # Modified: Replaced print with logging.warning
                    # logging.warning(f"Warning: Dimension mismatch for frame {frame_id}: "
                    #                 f"features {feature_map.shape[1:]} vs labels {gt_labels.shape}, resizing labels...")

                    # Resize labels to match feature map
                    gt_labels = F.interpolate(
                        torch.tensor(gt_labels[np.newaxis, np.newaxis, ...], dtype=torch.float32),
                        size=feature_map.shape[1:],
                        mode='nearest'
                    ).to(torch.int32).squeeze().numpy()
                
                # Compute predictions
                predictions = compute_pixel_predictions(feature_map, text_features, device)
                
                # Update confusion matrix
                update_confusion_matrix_fast(confusion_mat, gt_labels, predictions, num_classes)
                
                # Update counters
                valid_pixels = (gt_labels != -1).sum()
                total_pixels += valid_pixels
                processed_frames += 1
                
            except Exception as e:
                # Modified: Replaced print with logging.error
                logging.error(f"Error processing frame {frame_id} in scene {scene_id}: {e}")
                continue
    
    logging.info(f"\nProcessed {processed_frames} frames with {total_pixels:,} valid pixels")
    
    # Compute metrics
    ignore_classes = config.get('ignore_classes', [])
    metrics = compute_metrics(confusion_mat, ignore_classes)
    
    # Print results
    logging.info(f"\n=== Overall Results ===")
    logging.info(f"mIoU: {metrics['miou']:.4f}")
    logging.info(f"mAcc: {metrics['macc']:.4f}")
    
    if ignore_classes and metrics['fg_miou'] is not None:
        logging.info(f"\n=== Results Excluding Ignore Classes {ignore_classes} ===")
        logging.info("Excluded class names:")
        for idx in ignore_classes:
            if idx < len(class_names):
                logging.info(f"  {idx}: {class_names[idx]}")
        logging.info(f"Foreground mIoU: {metrics['fg_miou']:.4f}")
        logging.info(f"Foreground mAcc: {metrics['fg_macc']:.4f}")
    
    # Print per-class results
    logging.info(f"\n=== Per-Class Results ===")
    logging.info(f"{'Class':<30} {'IoU':>8} {'Acc':>8}")
    logging.info("-" * 48)
    
    for c in range(num_classes):
        if not np.isnan(metrics['iou_array'][c]):
            class_name = class_names[c] if c < len(class_names) else f"Class_{c}"
            logging.info(f"{class_name:<30} {metrics['iou_array'][c]:>8.4f} {metrics['acc_array'][c]:>8.4f}")

def main():
    parser = argparse.ArgumentParser(description="Zero-shot 2D semantic segmentation evaluation")
    parser.add_argument('--model_name', type=str, default='lseg',
                        help='Model name (siglip2 or clip or open_clip, lseg)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--log_dir', type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/logs', 
                        help='Directory to save log files')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    method_name = '3dgsback'
    log_filename = log_dir / f"{method_name}_2d_eval_{args.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Keep output clean, just the message
        handlers=[
            logging.FileHandler(log_filename), # Writes to file
            logging.StreamHandler()            # Writes to console
        ]
    )

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    if args.model_name == "lseg":
        import clip
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
        model = net.clip_pretrained.encode_text
        tokenizer = clip.tokenize
        model_type = "lseg"

    else:
        # Modified: Use logging before raising an error
        error_msg = f"Unsupported model name: {args.model_name}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Configuration for each dataset
    configs = {
        # 'scannet20': {
        #     'split_path': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/splits/scannet_mini_val.txt",
        #     'feature_base': "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/original_data",
        #     'label_base': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/2d_semseg_remapped_labels/scannet_mini_val/",
        #     'label_subfolder': "label-filt_20",
        #     'label_file': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/label20.txt",
        #     'ignore_classes': [19] 
        # },
        # 'scannet200': {
        #     'split_path': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/splits/scannet_mini_val.txt",
        #     'feature_base': "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/original_data",
        #     'label_base': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/2d_semseg_remapped_labels/scannet_mini_val/",
        #     'label_subfolder': "label-filt_200",
        #     'label_file': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/label200.txt",
        #     'ignore_classes': []
        # },
        'scannetpp100': {
            'split_path': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/splits/scannetpp_mini_val.txt",
            'feature_base': "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/",
            'label_base': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/2d_semseg_remapped_labels/scannetpp_mini_val",
            'label_subfolder': "semantics_100",
            'label_file': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/metadata/scannetpp_semseg_top100.txt",
            'ignore_classes': []
        }
    }
    
    # Evaluate each dataset
    for dataset_name, config in configs.items():
        # Check if label file exists
        if not os.path.exists(config['label_file']):
            logging.warning(f"\nWarning: Label file not found for {dataset_name}: {config['label_file']}")
            logging.warning(f"Skipping {dataset_name}")
            continue
        
        evaluate_dataset(dataset_name, config, model, tokenizer, device, model_type)

if __name__ == "__main__":
    main()