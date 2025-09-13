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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..") # noqa

from plyfile import PlyData
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import cKDTree as KDTree
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
KITTI_COLORS = generate_distinct_colors(37)



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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--val_split_path", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/kitti360_mini_val.txt")
    argparser.add_argument("--preprocessed_root", type=str, default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_chunk_preprocessed_gs/test")
    argparser.add_argument("--gs_root", type=str, default="/srv/beegfs-benderdata/scratch/qimaqi_data/data/neurips_2025/kitti_360_subset_chunk/")
    argparser.add_argument("--nn_num", type=int, default=25, help="Number of nearest neighbors to consider")
    argparser.add_argument("--print_class_iou", action="store_true")
    argparser.add_argument("--ignore_classes", nargs='+', default=[])
    # argparser.add_argument("--model_spec", type=str, default="siglip2-base-patch16-512")
    argparser.add_argument("--save_results", action="store_true", help="Save results to a file")
    argparser.add_argument("--downsample_ratio", default=-1, type=float, help="Downsample ratio for the point cloud to speed up the evaluation")
    argparser.add_argument("--label_path", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/metadata/kitti_label_37.txt")
    args = argparser.parse_args()

    val_split_path = args.val_split_path or       "/home/yli7/projects/yue/language_feat_exps/splits/holicity_mini_val.txt"
    preprocessed_root = args.preprocessed_root or "/home/yli7/scratch/datasets/ptv3_preprocessed/holicity"
    gs_root = args.gs_root or                     "/home/yli7/scratch/datasets/gaussian_world/outputs/ludvig/holicity" 
    langfeat_root = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/kitti360/2013_05_28_drive_0000_sync"
    label_path = args.label_path or "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/metadata/kitti_label_37.txt"
    nn_num = args.nn_num or 25
    args.print_class_iou = True
    args.ignore_classes = ["sky", "unknown construction", "unknown vehicle", "unknown object"]
    model_name = "clip"

    # Extract the 3DGS root folder name for the log filename
    gs_folder_name = os.path.basename(os.path.normpath(gs_root))
    
    # Setup for results logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"results_{gs_folder_name}_nn_num_{nn_num}.txt"
    log_path = os.path.join("logs", log_filename)
    
    # Capture all printed output
    stdout_original = sys.stdout
    results_capture = []
    
    class CaptureOutput:
        def write(self, text):
            results_capture.append(text)
            stdout_original.write(text)
        def flush(self):
            stdout_original.flush()
    
    sys.stdout = CaptureOutput()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use_siglip_probabilities = True

    # Load validation scenes
    scene_ids = load_scene_list(val_split_path)
    print(f"Using {model_name.upper()} language features from {langfeat_root}")
    print(f"Found {len(scene_ids)} validation scenes.")
    print(f"use nn_num: {nn_num}")  

    # ------------------------------
    # 2) Load CLIP model (once)
    # ------------------------------

    benchmarks = {
        "Kitti360": args.label_path,
    }

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

    clip_text_encoder = net.clip_pretrained.encode_text

    # ---- 3.1 Load label names & encode text ----
    with open(args.label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]

    print("prompt_list", prompt_list) 
    with torch.no_grad():
        text_tokens = clip.tokenize(prompt_list)
        text_feat = clip_text_encoder(text_tokens.to(device))  # (C, 512)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1)
        text_feat = text_feat.cpu()
    
    text_feat_lseg = text_feat.clone()
    text_feat_lseg = text_feat_lseg.numpy()
    print("text_feat_lseg shape: ", text_feat_lseg.shape)
    np.save(os.path.join("./kitti360_text_embeddings_lseg_clip.npy"), text_feat_lseg)
    num_classes = len(label_names)


    # text_feat = prepare_text_features(holicity_label_prompt)

    # Initialize metrics for both benchmarks
    benchmarks = [
        {
            'name': 'kitti360',
            'text_feat': text_feat,
            'class_labels': label_names,
            'confusion_mat': np.zeros((len(label_names), len(label_names)), dtype=np.int64),
            'fn_ignore': np.zeros(len(label_names), dtype=np.int64),
            'total_points': 0,
            'top1_correct': 0,
        },
    ]

    # Process each scene
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        print(f"\nProcessing scene {scene_id}...")
        scene_folder = os.path.join(preprocessed_root,  scene_id)
        # if not os.path.exists(scene_folder):
        #     scene_folder = os.path.join(preprocessed_root, "test", scene_id)
        # if not os.path.exists(scene_folder):
        #     raise ValueError(f"Scene {scene_id} not found in {preprocessed_root}")

        try:
            coord = np.load(os.path.join(scene_folder, "pc_coord.npy"))
            segment = np.load(os.path.join(scene_folder, "pc_segment.npy"))
            print("segment", segment.min(), segment.max())
            # print(f"Evaluating {scene_id} with {len(coord)} points...")
            # print(f"Segment unique values and counts: {np.unique(segment, return_counts=True)}")
        except:
            raise ValueError(f"Error loading data for scene {scene_id} in {scene_folder}")
        
        if args.downsample_ratio > 0:
            downsample_ratio = args.downsample_ratio
            num_points = len(coord)
            downsampled_indices = np.random.choice(num_points, int(num_points * downsample_ratio), replace=False)
            coord = coord[downsampled_indices]
            segment = segment[downsampled_indices]
            # print(f"Downsampled {scene_id} to {len(coord)} points...")

        # gs_path = os.path.join(gs_root, scene_id, model_name, "gaussians.ply")
        # if not os.path.exists(gs_path):
        #     raise ValueError(f"Error loading Gaussian data for scene {scene_id} in {gs_path}")
        
        # gauss_xyz, _ = read_ply_file_3dgs(gs_path)
        # langfeat_path = os.path.join(langfeat_root, scene_id, model_name, "features.npy")
        # if not os.path.isfile(langfeat_path):
        #     print(f"[Warning] Language feature not found at {langfeat_path}")
        #     continue
        # gauss_lang_feat = torch.from_numpy(np.load(langfeat_path)).float() # (G, 512)
        scene_id_only = scene_id[5:]
        xyz_path = os.path.join(langfeat_root, scene_id_only, "xyz_lseg_188_704.npy")
        langfeat_path = os.path.join(langfeat_root, scene_id_only, "features_lseg_188_704.pt")
        # pred_feat_path = os.path.join(args.results_root, scene_id, "features_lseg_584_876.pt")
        # xyz_path = os.path.join(args.results_root, scene_id, "xyz_lseg_584_876.npy")

        if not os.path.exists(langfeat_path):
            print(f"[Warning] Language feature not found at {langfeat_path}")
            continue
    
        gauss_lang_feat = torch.load(langfeat_path).cpu().float()  # (G, 512)

        gauss_xyz = np.load(xyz_path)  # (G, 3)


        for bench in benchmarks:
            # Select current benchmark data
            current_segment = segment
            text_feat = bench['text_feat']
            class_labels = bench['class_labels']
            num_classes = len(class_labels)

            # Each element in current_segment is an array of valid GT class indices for that point
            gt_labels = [row[row >= 0] for row in current_segment]
            valid_mask = np.array([len(labels) > 0 for labels in gt_labels], dtype=bool)
            if not np.any(valid_mask):
                continue

            xyz_val = coord[valid_mask]
            gt_val = [gt_labels[i] for i in np.where(valid_mask)[0]]
            # For simplicity, pick the "first" ground-truth label as the canonical GT
            gt_first = [g[0] for g in gt_val]

            # Compute predictions for the Gaussian samples
            batch_size = 128000
            gauss_labels = []
            for i in range(0, len(gauss_lang_feat), batch_size):
                batch_feat = gauss_lang_feat[i:i+batch_size].to(device)
                batch_pred = compute_relevancy_scores(
                    batch_feat, text_feat, device, bench['name']
                )
                gauss_labels.append(batch_pred)
            gauss_labels = np.concatenate(gauss_labels, axis=0)

            # KDTree search and voting
            kd_tree = KDTree(gauss_xyz)
            _, nn_indices = kd_tree.query(xyz_val, k=nn_num)
            neighbor_labels = gauss_labels[nn_indices].squeeze(-1)

            # Voting logic
            top1_preds = []
            for neighbors in neighbor_labels:
                valid_neighbors = neighbors[neighbors != -1]  # ignore "no confident prediction"
                if len(valid_neighbors) == 0:
                    top1_preds.append(-1)
                else:
                    counts = np.bincount(valid_neighbors)
                    top1_preds.append(np.argmax(counts))
            top1_preds = np.array(top1_preds)

            # Due to the flattening of the labels of Holicity, cars and pedestrians are not labeled
            _, nn_idx_all = kd_tree.query(coord,  k=nn_num)             # (N,)
            neighbor_labels_all = gauss_labels[nn_idx_all].squeeze(-1)

            top1_preds_all = []
            for neighbors in neighbor_labels_all:
                valid_neighbors = neighbors[neighbors != -1]  # ignore "no confident prediction"
                if len(valid_neighbors) == 0:
                    top1_preds_all.append(-1)
                else:
                    counts = np.bincount(valid_neighbors)
                    top1_preds_all.append(np.argmax(counts))
            top1_preds_all = np.array(top1_preds_all)
            # print("top1_preds_all", top1_preds_all)

            save_fname = f"kitti360_backproject_lseg_{scene_id}_semseg_pred.npy"
            save_path = os.path.join(langfeat_root, scene_id, save_fname)
            #os.path.join(f"output/ludvig/{split_name}", save_fname)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, top1_preds_all.astype(np.int32))

            # colorize the predictions 
            colorized_pred = KITTI_COLORS[top1_preds_all]
            open3d_vis = o3d.geometry.PointCloud()
            open3d_vis.points = o3d.utility.Vector3dVector(coord)
            open3d_vis.colors = o3d.utility.Vector3dVector(colorized_pred)
            save_vis_path = os.path.join(langfeat_root, scene_id, f"kitti360_backproject_lseg_{scene_id}_semseg_pred.ply")
            o3d.io.write_point_cloud(save_vis_path, open3d_vis)
            print("Save visualization to", save_vis_path)
            

            # Update metrics
            bench['total_points'] += len(gt_val)
            for i, (g, pred) in enumerate(zip(gt_val, top1_preds)):    
                gt_c = gt_first[i]
                # Update confusion matrix
                if pred == -1:
                    bench['fn_ignore'][gt_c] += 1
                else:
                    if 0 <= gt_c < num_classes and 0 <= pred < num_classes:
                        bench['confusion_mat'][gt_c, pred] += 1

                # Update top-1 "correct" if the predicted label is in the GT set
                if pred in g:
                    bench['top1_correct'] += 1

    # Compute and print results for each benchmark
    for bench in benchmarks:
        print(f"\n=== Results for {bench['name'].upper()} ===")
        num_classes = len(bench['class_labels'])
        cm = bench['confusion_mat']
        fn_ignore = bench['fn_ignore']

        # Global accuracy
        global_acc = bench['top1_correct'] / bench['total_points'] if bench['total_points'] > 0 else 0
        
        # Arrays to store IoU and Acc for each class (indexed by class ID)
        iou_array = np.full(num_classes, np.nan, dtype=float)
        acc_array = np.full(num_classes, np.nan, dtype=float)

        # Compute per-class metrics
        for c in range(num_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp + fn_ignore[c]
            # If no points of this class exist at all, skip
            if (tp + fn) == 0:
                continue

            acc = tp / (tp + fn)  # class accuracy
            denom = (tp + fp + fn)
            iou = (tp / denom) if denom > 0 else 0.0

            iou_array[c] = iou
            acc_array[c] = acc

        # Mean class accuracy / IoU over classes that are not NaN
        valid_mask = ~np.isnan(iou_array)
        valid_acc_vals = acc_array[valid_mask]
        valid_iou_vals = iou_array[valid_mask]
        macc = np.mean(valid_acc_vals) if len(valid_acc_vals) > 0 else 0
        miou = np.mean(valid_iou_vals) if len(valid_iou_vals) > 0 else 0

        # Foreground metrics: exclude user-specified classes (e.g. wall, floor, ceiling)
        excluded_indices = [
            i for i, name in enumerate(bench['class_labels']) if name in args.ignore_classes
        ] if args.ignore_classes else []
        print(f"Excluded indices: {excluded_indices}")
        # We only consider classes that are valid and not in the excluded set
        fg_mask = valid_mask.copy()
        fg_mask[excluded_indices] = False

        fg_acc_vals = acc_array[fg_mask]
        fg_iou_vals = iou_array[fg_mask]
        fg_macc = np.mean(fg_acc_vals) if len(fg_acc_vals) > 0 else 0
        fg_miou = np.mean(fg_iou_vals) if len(fg_iou_vals) > 0 else 0

        print(f"Global Accuracy: {global_acc:.4f}")
        print(f"Mean Class Accuracy: {macc:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"Foreground excluding classes: {args.ignore_classes}")
        print(f"Foreground mIoU: {fg_miou:.4f}")
        print(f"Foreground mAcc: {fg_macc:.4f}")

        # Print per-class IoU if requested
        if args.print_class_iou:
            print("\nPer-class IoU:")
            for c in range(num_classes):
                if not np.isnan(iou_array[c]):
                    class_name = bench['class_labels'][c]
                    print(f"{class_name:<20}: {iou_array[c]:.4f}")

    # Restore stdout and save results to file
    sys.stdout = stdout_original
    results_str = ''.join(results_capture)
    split_name = val_split_path.split("/")[-1].split(".")[0]
    log_path = f"logs/3dgs_back_3d_semseg_eval_{split_name}_{model_name}.txt"
    save_results_to_file(log_path, results_str, args)

if __name__ == "__main__":
    main()