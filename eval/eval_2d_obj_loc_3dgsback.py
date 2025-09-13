"""
Zero-shot 2D object localization evaluation script for ScanNet and ScanNet++
Evaluates rendered 2D feature maps against ground truth object bounding boxes and segmentation masks
"""

import os
import json
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
import cv2
import sys
sys.path.append("..")

def load_scene_list(split_path):
    """Load scene IDs from split file."""
    with open(split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if line.strip()]
    return scene_ids

def load_object_annotations(json_path):
    """Load object annotations from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

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


@torch.no_grad()
def compute_object_location(feature_map, text_feature, device, use_smoothing=True, smooth_kernel_size=30):
    """
    Compute object location using maximum similarity with optional smoothing.
    
    Args:
        feature_map: [C, H, W] torch tensor
        text_feature: [C] torch tensor for single object category
        device: torch device
        use_smoothing: whether to apply smoothing to activation map
        smooth_kernel_size: size of smoothing kernel
    
    Returns:
        location: (x, y) tuple of predicted location (width, height coordinates)
        activation_map: [H, W] numpy array of activation values
    """
    C, H, W = feature_map.shape
    
    # Move to device
    feature_map = feature_map.to(device, non_blocking=True)
    text_feature = text_feature.to(device, non_blocking=True)
    
    # Normalize feature map
    feature_map = feature_map / (feature_map.norm(dim=0, keepdim=True) + 1e-10)
    
    # Compute similarity scores
    # Reshape for matrix multiplication
    features_flat = feature_map.reshape(C, H * W).t()  # [H*W, C]
    similarity = torch.matmul(features_flat, text_feature)  # [H*W]
    activation_map = similarity.reshape(H, W).cpu().numpy()
    
    if use_smoothing:
        # Apply box filter smoothing like LangSplat
        kernel = np.ones((smooth_kernel_size, smooth_kernel_size)) / (smooth_kernel_size ** 2)
        smoothed_map = cv2.filter2D(activation_map, -1, kernel)
        
        # Find maximum in smoothed map
        max_loc = np.unravel_index(np.argmax(smoothed_map), smoothed_map.shape)
        y, x = max_loc  # Note: numpy returns (row, col) = (height, width)
        
        # Combine smoothed and original maps (optional, for visualization)
        # activation_map = 0.5 * (smoothed_map + activation_map)
    else:
        # Find maximum in original map
        max_loc = np.unravel_index(np.argmax(activation_map), activation_map.shape)
        y, x = max_loc
    
    return (x, y), activation_map

def check_bbox_hit(location, bbox):
    """Check if location falls within bounding box."""
    x, y = location
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

def check_segmentation_hit(location, segmentation_pixels):
    """Check if location falls within segmentation mask."""
    x, y = location
    # segmentation_pixels is a list of [x, y] coordinates
    for pixel in segmentation_pixels:
        if int(pixel[0]) == x and int(pixel[1]) == y:
            return True
    return False

def evaluate_dataset(dataset_name, config, model, tokenizer, device, model_type, 
                    use_smoothing=True, smooth_kernel_size=30):
    """Evaluate a single dataset configuration."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating {dataset_name}")
    logging.info(f"{'='*60}")
    
    # Load scene list
    scene_ids = load_scene_list(config['split_path'])
    logging.info(f"Found {len(scene_ids)} scenes")
    # scene_ids = scene_ids[:2]
    
    # Initialize counters
    total_objects = 0
    bbox_hits = 0
    segmentation_hits = 0
    
    # Per-category statistics
    category_stats = defaultdict(lambda: {
        'total': 0,
        'bbox_hits': 0,
        'seg_hits': 0
    })
    
    # Process each scene
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        # Get feature and label directories
        # print("scene_id", scene_id)
        # feature_dir = Path(config['feature_base']) / scene_id / "test" / "renders"
        label_dir = Path(config['obj_loc_label_base']) / scene_id
        
        # Get feature and label directories
        feature_dir = Path(config['feature_base']) / scene_id  / "3dgsback_render_feats" 

        if not feature_dir.exists():
            logging.warning(f"Warning: Feature directory not found for scene {scene_id}")
            continue
        
        if not label_dir.exists():
            logging.warning(f"Warning: Label directory not found for scene {scene_id}")
            continue
        
        # Get all feature files
        feature_files = sorted(feature_dir.glob("*.pth"))
        
        for feat_file in tqdm(feature_files, desc=f"Processing views in {scene_id}", leave=False):
            frame_id = feat_file.stem
            frame_id = frame_id.split('_')[0]  # Extract frame ID from filename
            
            # Construct label file path
            label_file = label_dir / f"frame_{frame_id}.json"
            
            if not label_file.exists():
                logging.warning(f"Info: Label file {label_file} not found in scene {scene_id}") # some test views may not have labels
                continue
            
            # try:
            # Load feature map and annotations
            feature_map = torch.load(feat_file, map_location='cpu', weights_only=True)  # [C, H, W]
            feature_map = feature_map.float()
            feature_map = feature_map.permute(2, 0, 1)  # Convert to [H, W, C] format
            # feature_map = np.load(feat_file)  # [ H, W, C]

            annotations = load_object_annotations(label_file)
            
            # Get image dimensions from annotations
            img_width = annotations['info']['width']
            img_height = annotations['info']['height']
            
            # Check if feature map dimensions match image dimensions
            if feature_map.shape[1] != img_height or feature_map.shape[2] != img_width:
                # Resize feature map to match image dimensions
                feature_map = F.interpolate(
                    feature_map.unsqueeze(0),
                    size=(img_height, img_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Process each object in the frame
            for obj in annotations['objects']:
                category = obj['category']
                bbox = obj['bbox']
                segmentation = obj['segmentation']
                
                # Prepare text prompt
                text_prompt = f"this is a {category}"
                text_feature = prepare_text_features([text_prompt], model, tokenizer, device, model_type)
                text_feature = text_feature.squeeze(0)  # Remove batch dimension
                text_feature = text_feature.float()  # Ensure float type
                
                # Find object location
                location, activation_map = compute_object_location(
                    feature_map, text_feature, device, 
                    use_smoothing=use_smoothing,
                    smooth_kernel_size=smooth_kernel_size
                )
                
                # Check if location hits bbox
                bbox_hit = check_bbox_hit(location, bbox)
                seg_hit = check_segmentation_hit(location, segmentation)
                
                # Update counters
                total_objects += 1
                if bbox_hit:
                    bbox_hits += 1
                if seg_hit:
                    segmentation_hits += 1
                
                # Update category statistics
                category_stats[category]['total'] += 1
                if bbox_hit:
                    category_stats[category]['bbox_hits'] += 1
                if seg_hit:
                    category_stats[category]['seg_hits'] += 1
                
            # except Exception as e:
            #     logging.error(f"Error processing frame {frame_id} in scene {scene_id}: {e}")
            #     continue
    
    # Compute accuracies
    bbox_accuracy = bbox_hits / total_objects if total_objects > 0 else 0
    seg_accuracy = segmentation_hits / total_objects if total_objects > 0 else 0
    
    # Print results
    logging.info(f"\n=== Overall Results ===")
    logging.info(f"Total objects evaluated: {total_objects}")
    logging.info(f"Bounding Box-based Accuracy: {bbox_accuracy:.4f} ({bbox_hits}/{total_objects})")
    logging.info(f"Segmentation-based Accuracy: {seg_accuracy:.4f} ({segmentation_hits}/{total_objects})\n")
    
    # Sort categories by total count
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for category, stats in sorted_categories:
        total = stats['total']
        bbox_acc = stats['bbox_hits'] / total if total > 0 else 0
        seg_acc = stats['seg_hits'] / total if total > 0 else 0
        logging.info(f"{category:<30} {total:>8} {bbox_acc:>10.4f} {seg_acc:>10.4f}")
    
    return {
        'total_objects': total_objects,
        'bbox_accuracy': bbox_accuracy,
        'seg_accuracy': seg_accuracy,
        'category_stats': dict(category_stats)
    }

def main():
    parser = argparse.ArgumentParser(description="Zero-shot 2D object localization evaluation")
    parser.add_argument('--model_name', type=str, default='lseg',
                        help='Model name (siglip2 or clip)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--log_dir', type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/eval/log',
                        help='Directory to save log files')
    parser.add_argument('--use_smoothing', action='store_true', default=True,
                        help='Whether to use smoothing for activation maps')
    parser.add_argument('--smooth_kernel_size', type=int, default=30,
                        help='Size of smoothing kernel')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    method_name = '3dgsback'
    log_filename = log_dir / f"{method_name}_obj_loc_eval_{args.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Smoothing: {'Enabled' if args.use_smoothing else 'Disabled'}")
    if args.use_smoothing:
        logging.info(f"Smooth kernel size: {args.smooth_kernel_size}")
    
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
    split = "tiny_val"  # Use mini_val for evaluation
    logging.info(f"\nUsing split: {split}")
    configs = {
        'scannet': {
            'split_path': f"/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/splits/scannet_mini_val.txt",
            'feature_base': "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/original_data/",
            'obj_loc_label_base': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/obj_loc_labels/scannet_mini_val",
        },
        # 'scannetpp': {
        #     'split_path': f"/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/splits/scannetpp_mini_val.txt",
        #     'feature_base': "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/",
        #     'obj_loc_label_base': "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/2D_eval/obj_loc_labels/scannetpp_mini_val",
            
        # }
    }
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, config in configs.items():
        # Check if label directory exists
        if not os.path.exists(config['obj_loc_label_base']):
            logging.warning(f"\nWarning: Object localization label directory not found for {dataset_name}: {config['obj_loc_label_base']}")
            logging.warning(f"Skipping {dataset_name}")
            continue
        
        results = evaluate_dataset(
            dataset_name, config, model, tokenizer, device, model_type,
            use_smoothing=args.use_smoothing,
            smooth_kernel_size=args.smooth_kernel_size
        )
        all_results[dataset_name] = results
    
    logging.info(f"\n{'='*60}")
    logging.info("Summary of All Datasets")
    logging.info(f"{'='*60}")
    
    total_objects_all = sum(r['total_objects'] for r in all_results.values())
    total_bbox_hits = sum(r['bbox_accuracy'] * r['total_objects'] for r in all_results.values())
    total_seg_hits = sum(r['seg_accuracy'] * r['total_objects'] for r in all_results.values())
    
    if total_objects_all > 0:
        overall_bbox_acc = total_bbox_hits / total_objects_all
        overall_seg_acc = total_seg_hits / total_objects_all
        
        logging.info(f"Total objects across all datasets: {total_objects_all}")
        logging.info(f"Overall Bounding Box Accuracy: {overall_bbox_acc:.4f}")
        logging.info(f"Overall Segmentation Accuracy: {overall_seg_acc:.4f}")

if __name__ == "__main__":
    main()