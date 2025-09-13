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
    parser.add_argument("--val_split_path", type=str, default="/insait/qimaqi/workspace/3dgs-gradient-backprojection-benchmark/splits/scannet_mini_val.txt") # scannet_mini_val  scannet_val
    parser.add_argument("--preprocessed_root", type=str, default="/insait/GSWorld/downloaded_datasets/scannet/scannet_preprocess/val")
    parser.add_argument("--results_root", type=str, default="/insait/qimaqi/workspace/3dgs-gradient-backprojection-benchmark/results/scannet/")
    # parser.add_argument("--xyz_root", type=str, default="/insait/qimaqi/workspace/LangSplat_Benchmark/output_scannet_bak/scene0011_00/feature_level_3/xyz.npy")
    # parser.add_argument("--langfeat_root", type=str, default="/insait/qimaqi/workspace/LangSplat_Benchmark/output_scannet_bak/scene0011_00/feature_level_3/features.npy")
    parser.add_argument("--label20_path", type=str, default="/insait/GSWorld/downloaded_datasets/scannet/metadata/semantic_benchmark/label20.txt")
    parser.add_argument("--label200_path", type=str, default="/insait/GSWorld/downloaded_datasets/scannet/metadata/semantic_benchmark/label200.txt")
    parser.add_argument("--use_dot_similarity", action="store_true", help="If set, use plain CLIP dot similarity instead of ratio scoring.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_dot_similarity = True  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # 1) Load validation scenes
    # ------------------------------
    scene_ids = load_scene_list(args.val_split_path)
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
    net.load_state_dict(torch.load("/insait/qimaqi/workspace/3dgs-gradient-backprojection-benchmark/checkpoints/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text

    # prompts = [prompt] + neg_prompt.split(";")

    # prompt = clip.tokenize(prompts)
    # prompt = prompt.cuda()

    # text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
    # text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)


    # Canonical phrases stay fixed
    canonical_phrases = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        canon_tokens = clip.tokenize(canonical_phrases)
        canon_feat = clip_text_encoder(canon_tokens.to(device))
        # canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = torch.nn.functional.normalize(canon_feat, dim=1)
        canon_feat = canon_feat.cpu()

    benchmarks = {
        "ScanNet20": args.label20_path,
        "ScanNet200": args.label200_path,
    }

    ignore_classes = [""]#["wall", "floor", "ceiling"]  # for foreground metrics

    for bench_name, label_path in benchmarks.items():
        print(f"\n===== Evaluating {bench_name} =====")
        # ---- 3.1 Load label names & encode text ----
        with open(label_path, "r") as f:
            label_names = [line.strip() for line in f if len(line.strip()) > 0]
        prompt_list = ["this is a " + name for name in label_names]

        with torch.no_grad():
            text_tokens = clip.tokenize(prompt_list)
            text_feat = clip_text_encoder(text_tokens.to(device))  # (C, 512)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu()

        num_classes = len(label_names)
        confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Build ignore mask once
        ignore_mask = np.array([name in ignore_classes for name in label_names], dtype=bool)

        # ---- 3.2 Loop over scenes ----
        for scene_id in tqdm(scene_ids, desc="Scene", dynamic_ncols=True):
            # Paths to data
            scene_preproc_folder = os.path.join(args.preprocessed_root, scene_id)
            # scene_preproc_folder = os.path.join(args.preprocessed_root, "val", scene_id)
            if not os.path.isdir(scene_preproc_folder):
                print(f"[Warning] Preprocessed folder not found: {scene_preproc_folder}")
                continue

            coord_path = os.path.join(scene_preproc_folder, "coord.npy")
            segment_path = os.path.join(scene_preproc_folder, "segment20.npy" if bench_name == "ScanNet20" else "segment200.npy")
            if not (os.path.isfile(coord_path) and os.path.isfile(segment_path)):
                print(f"[Warning] Missing coord.npy or segment.npy at {scene_preproc_folder}")
                continue

            coord = np.load(coord_path)               # (N, 3)
            segment = np.load(segment_path)           # (N, 3) first col => label index
            labeled_gt = segment#[:, 0]                
            valid_mask = labeled_gt >= 0
            if valid_mask.sum() == 0:
                continue

            xyz_val = coord[valid_mask]
            gt_val = labeled_gt[valid_mask].astype(np.int64)

            pred_feat_path = os.path.join(args.results_root, scene_id, "features_lseg_480_640.pt")
            if not os.path.exists(pred_feat_path):
                print(f"[Warning] Predicted features not found at {pred_feat_path}")
                continue
            gauss_lang_feat = torch.load(pred_feat_path).cpu().float()  # (G, 512)
            xyz_path = os.path.join(args.results_root, scene_id, "xyz_lseg_480_640.npy")
            gauss_xyz = np.load(xyz_path)  # (G, 3)

            assert gauss_lang_feat.shape[0] == gauss_xyz.shape[0], f"Mismatch in number of features: {gauss_lang_feat.shape[0]} vs {gauss_xyz.shape[0]}"

            # ---- 3.2.b Load 3DGS & CLIP feats ----
            # scene_3dgs_folder = os.path.join(args.gs_root, scene_id)
            # ply_path = os.path.join(scene_3dgs_folder, "clip", "gaussians.ply")
            # if not os.path.isfile(ply_path):
            #     print(f"[Warning] 3DGS .ply not found for scene {scene_id}")
            #     continue
            # gauss_xyz, _ = read_ply_file_3dgs(ply_path)

            # langfeat_path = os.path.join(args.langfeat_root, scene_id, "clip", "features.npy")
            # if not os.path.isfile(langfeat_path):
            #     print(f"[Warning] Language feature not found at {langfeat_path}")
            #     continue
            # gauss_lang_feat = torch.from_numpy(np.load(langfeat_path)).float() # (G, 512)

            norms = gauss_lang_feat.norm(dim=1)
            keep_mask_gs = (norms > 0)
            gauss_xyz = gauss_xyz[keep_mask_gs.numpy()]
            gauss_lang_feat = gauss_lang_feat[keep_mask_gs]
            if gauss_xyz.shape[0] == 0:
                print(f"[Warning] All 3DGS zero feats in {scene_id}")
                continue

            # ---- 3.2.c KD‑Tree NN search ----
            kd_tree = KDTree(gauss_xyz)
            _, nn_idx = kd_tree.query(xyz_val)
            nn_lang_feat = gauss_lang_feat[nn_idx]

         

            # ---- 3.2.d Predict labels ----
            pred_label = compute_relevancy_scores(
                nn_lang_feat,
                text_feat,
                canon_feat,
                device=device,
                use_dot_similarity=args.use_dot_similarity,
            )

            # ---- 3.2.e Accumulate confusion ----
            for gt_c, pr_c in zip(gt_val, pred_label):
                if gt_c < num_classes and pr_c < num_classes:  # guard against idx mismatch
                    confusion_mat[gt_c, pr_c] += 1

        # ------------------------------
        # 4) Compute metrics
        # ------------------------------
        ious = []
        per_class_acc = []
        gt_class_counts = np.sum(confusion_mat, axis=1)

        for c in range(num_classes):
            tp = confusion_mat[c, c]
            fn = gt_class_counts[c] - tp
            fp = np.sum(confusion_mat[:, c]) - tp
            denom = tp + fp + fn
            iou_c = tp / denom if denom > 0 else 0.0
            ious.append(iou_c)

            acc_c = tp / gt_class_counts[c] if gt_class_counts[c] > 0 else 0.0
            per_class_acc.append(acc_c)

        valid_mask = gt_class_counts > 0
        mean_iou = np.mean(np.array(ious)[valid_mask]) if valid_mask.any() else 0.0
        mean_class_acc = np.mean(np.array(per_class_acc)[valid_mask]) if valid_mask.any() else 0.0

        total_correct = np.trace(confusion_mat)
        total_count = confusion_mat.sum()
        global_acc = total_correct / (total_count + 1e-12)

        # Foreground metrics (exclude ignore classes)
        fg_mask = valid_mask & (~ignore_mask)
        fg_miou = np.mean(np.array(ious)[fg_mask]) if fg_mask.any() else 0.0
        fg_macc = np.mean(np.array(per_class_acc)[fg_mask]) if fg_mask.any() else 0.0

        # ------------------------------
        # 5) Print final results
        # ------------------------------
        print("\n======== RESULTS ========")
        print("Per‑class IoU:")
        for c, name in enumerate(label_names):
            print(f"  {name:24s}: {ious[c]:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Global Accuracy: {global_acc:.4f}")
        print(f"Mean Class Accuracy: {mean_class_acc:.4f}")
        print(f"Foreground mIoU: {fg_miou:.4f}")
        print(f"Foreground mAcc: {fg_macc:.4f}")


if __name__ == "__main__":
    main()