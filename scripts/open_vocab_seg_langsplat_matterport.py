import os
import argparse
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
# import open_clip
from tqdm import tqdm
from lseg import LSegNet
import clip
import sys
from metadata.matterport3d import MATTERPORT_LABELS_21, MATTERPORT_LABELS_160

###################################
# 1. Basic I/O Utilities
###################################

def load_scene_list(val_split_path):
    """
    Reads a .txt file listing validation scenes (one per line).
    Returns a list of scene IDs (strings).
    """
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if len(line.strip()) > 0]
    return scene_ids


def read_ply_file_3dgs(file_path):
    """
    Reads the 3D Gaussian ply (e.g. point_cloud_30000.ply).
    Returns xyz and opacity.
    """
    ply_data = PlyData.read(file_path)
    vertex = ply_data["vertex"]
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    opacity = vertex["opacity"]
    xyz = np.stack([x, y, z], axis=-1)
    return xyz, opacity

###################################
# 2. CLIP Relevancy Scoring
###################################
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

@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,      # shape: (N, 512)
    text_feat: torch.Tensor,      # shape: (C, 512)
    canon_feat: torch.Tensor,     # shape: (K, 512)
    device: torch.device,
    use_dot_similarity: bool = False,
):
    """
    Computes predicted labels for each of the N language features using one of two
    methods:

    1. Ratio method (default, identical to original code):
       Score_c = min_i [ exp(lang . text_c) / (exp(lang . canon_i) + exp(lang . text_c)) ]
       where i indexes the canonical phrases.

    2. Dot‑similarity method (if ``use_dot_similarity`` is ``True``):
       Simply picks the text embedding with the highest CLIP dot similarity.

    Returns
    -------
    pred_label : ndarray, shape (N,)
        The predicted label indices in [0..C‑1].
    """

    # Move to device
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    canon_feat = canon_feat.to(device, non_blocking=True)

    # Fast path: plain dot‑product similarity
    lang_feat = lang_feat.to(text_feat.dtype)  # (N, 512)
    if use_dot_similarity:
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        pred_label = torch.argmax(dot_lang_text, dim=1)
        return pred_label.cpu().numpy()

    # Original ratio‑based relevancy score
    dot_lang_text = torch.matmul(lang_feat, text_feat.t())    # (N, C)
    dot_lang_canon = torch.matmul(lang_feat, canon_feat.t())  # (N, K)

    exp_lang_text = dot_lang_text.exp()    # (N, C)
    exp_lang_canon = dot_lang_canon.exp()  # (N, K)

    N, C = dot_lang_text.shape

    relevancy_scores = []
    for c_idx in range(C):
        text_c_exp = exp_lang_text[:, c_idx].unsqueeze(-1)  # (N,1)
        ratio_c = text_c_exp / (exp_lang_canon + text_c_exp)  # (N, K)
        score_c = torch.min(ratio_c, dim=1).values  # (N,)
        relevancy_scores.append(score_c)

    relevancy_matrix = torch.stack(relevancy_scores, dim=0).t()  # (N, C)
    pred_label = torch.argmax(relevancy_matrix, dim=1)           # (N,)
    return pred_label.cpu().numpy()

###################################
# 3. Main Evaluation Script
###################################

def parse_args():
    parser = argparse.ArgumentParser(description="Open‑Vocal 3DGS semantic evaluation")
    parser.add_argument("--val_split_path", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/matterport3d_test.txt")
    parser.add_argument("--preprocessed_root", type=str, default="/srv/beegfs02/scratch/qimaqi_data/data/matterport_pointcloud_preprocessed/")
    parser.add_argument("--results_root", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/matterport")
    parser.add_argument("--gs_root", type=str, default="/srv/beegfs02/scratch/qimaqi_data/data/matterport_all_val/mcmc_3dgs/")
    parser.add_argument("--nn_num", type=int, default=25)
    parser.add_argument("--label_path", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt")
    parser.add_argument("--use_dot_similarity", action="store_true", help="If set, use plain CLIP dot similarity instead of ratio scoring.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_dot_similarity = True  
    preprocessed_root = args.preprocessed_root
    gs_root = args.gs_root
    results_root = args.results_root
    nn_num = args.nn_num
    val_split_path = args.val_split_path
    model_name = 'lseg_minimal_clip'
    split_name = val_split_path.split("/")[-1].split(".")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # 1) Load validation scenes
    # ------------------------------
    scene_ids = ['q9vSo1VnCiC_05'] #load_scene_list(val_split_path)
    # scene_ids = [scene_ids[0]] 
    print("scene_ids: ", scene_ids)
    print(f"Found {len(scene_ids)} validation scenes.")

    # ------------------------------
    # 2) Load CLIP model (once)
    # ------------------------------
    # model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    # model = model.eval().to(device)
    # tokenizer = open_clip.get_tokenizer("ViT-B-16")


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


    text_prompts_21 = ["this is a " + label for label in MATTERPORT_LABELS_21]
    text_prompts_160 = ["this is a " + label for label in MATTERPORT_LABELS_160]
    # text_feat_path = "/home/yli7/scratch/datasets/holicity/metadata/holicity_text_embeddings_siglip2.pt"
    # text_feat_21 = torch.load(text_feat_path, map_location="cpu")

    # ------------------------------
    # 3) Evaluate 
    # ------------------------------
    args.print_class_iou = True
    args.ignore_classes = ['other furniture', 'wall', 'floor', 'ceiling']
    stdout_original = sys.stdout
    results_capture = []
    

    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts_21)
        text_feat = clip_text_encoder(text_tokens.to(device))  # (C, 512)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1)
        text_feat = text_feat.cpu()
    text_feat_21 = text_feat

    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts_160)
        text_feat = clip_text_encoder(text_tokens.to(device))  # (C, 512)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1)
        text_feat = text_feat.cpu()
    text_feat_160 = text_feat


    benchmarks = [
        {
            'name': 'matterport3d_semseg_21',
            'text_feat': text_feat_21,
            'class_labels': MATTERPORT_LABELS_21,
            'confusion_mat': np.zeros((len(MATTERPORT_LABELS_21), len(MATTERPORT_LABELS_21)), dtype=np.int64),
            'fn_ignore': np.zeros(len(MATTERPORT_LABELS_21), dtype=np.int64),
            'total_points': 0,
            'top1_correct': 0,
        }
        # {
        #     'name': 'matterport3d_semseg_160',
        #     'text_feat': text_feat_160,
        #     'class_labels': MATTERPORT_LABELS_160,
        #     'confusion_mat': np.zeros((len(MATTERPORT_LABELS_160), len(MATTERPORT_LABELS_160)), dtype=np.int64),
        #     'fn_ignore': np.zeros(len(MATTERPORT_LABELS_160), dtype=np.int64),
        #     'total_points': 0,
        #     'top1_correct': 0,
        # }
    ]

    # Process each scene
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        scene_folder = os.path.join(preprocessed_root, scene_id)
        # if not os.path.exists(scene_folder):
        #     scene_folder = os.path.join(preprocessed_root, "test", scene_id)
        # if not os.path.exists(scene_folder):
        #     scene_folder = os.path.join(preprocessed_root, "train", scene_id)
        # if not os.path.exists(scene_folder):
        #     raise ValueError(f"Scene {scene_id} not found in {preprocessed_root}")

        try:
            coord = np.load(os.path.join(scene_folder, "coord.npy"))
            segment21 = np.load(os.path.join(scene_folder, "segment.npy"))
            # segment160 = np.load(os.path.join(scene_folder, "segment_nyu_160.npy"))
        except:
            raise ValueError(f"Error loading data for scene {scene_id} in {scene_folder}")

        # gs_path = os.path.join(gs_root, scene_id, 'ckpts', "gaussians.ply")
        # if not os.path.exists(gs_path):
        #     raise ValueError(f"Error loading Gaussian data for scene {scene_id} in {gs_path}")
        
        # gauss_xyz, _ = read_ply_file_3dgs(gs_path)
        xyz_path = os.path.join(results_root, scene_id, "xyz_lseg_512_640.npy")
        langfeat_path = os.path.join(results_root, scene_id, "features_lseg_512_640.pt")

        # pred_feat_path = os.path.join(args.results_root, scene_id, "features_lseg_584_876.pt")
        # xyz_path = os.path.join(args.results_root, scene_id, "xyz_lseg_584_876.npy")

        if not os.path.exists(langfeat_path):
            continue
    
        # gauss_lang_feat = torch.load(pred_feat_path).cpu().float()  # (G, 512)

        gauss_xyz = np.load(xyz_path)  # (G, 3)


        if not os.path.isfile(langfeat_path):
            print(f"[Warning] Language feature not found at {langfeat_path}")
            continue
        # gauss_lang_feat = torch.from_numpy(np.load(langfeat_path)).float() # (G, 512)        
        gauss_lang_feat = torch.load(langfeat_path).cpu().float()  # (G, 512)


        for bench in benchmarks:
            # Select current benchmark data
            current_segment = segment21 # if bench['name'] == 'matterport3d_semseg_21'
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
                    batch_feat, text_feat, text_feat, device, True
                )
                gauss_labels.append(batch_pred)
            gauss_labels = np.concatenate(gauss_labels, axis=0)

            # KDTree search and voting
            kd_tree = KDTree(gauss_xyz)
            _, nn_indices = kd_tree.query(xyz_val, k=nn_num)
            neighbor_labels = gauss_labels[nn_indices]

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

            if True:
                _, nn_indices_all = kd_tree.query(coord, k=nn_num)          # (N, k)
                neighbor_labels_all = gauss_labels[nn_indices_all]

                full_preds = []
                for neighbors in neighbor_labels_all:
                    valid_neighbors = neighbors[neighbors != -1]
                    if len(valid_neighbors) == 0:
                        full_preds.append(-1)
                    else:
                        counts = np.bincount(valid_neighbors)
                        full_preds.append(np.argmax(counts))
                full_preds = np.array(full_preds, dtype=np.int32)         # (N,)

                save_fname = f"3dgs_back_project_{model_name}_{scene_id}_semseg_pred.npy"
                save_path = os.path.join(f"output/3dgs_back_project/{split_name}", save_fname)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, full_preds)

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
        ]
        print(f"Excluded indices: {excluded_indices}, classes: {args.ignore_classes}")
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
                    # only print class that presents during evaluation
                    class_name = bench['class_labels'][c]
                    print(f"{class_name:<20}: {iou_array[c]:.4f}")

    # Restore stdout and save results to file
    sys.stdout = stdout_original
    results_str = ''.join(results_capture)
    split_name = val_split_path.split("/")[-1].split(".")[0]
    log_path = f"logs/backprojection_3d_semseg_eval_{split_name}_{'clip'}.txt"
    save_results_to_file(log_path, results_str, args)



if __name__ == "__main__":
    main()