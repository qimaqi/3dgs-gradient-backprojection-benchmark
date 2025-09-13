import math
import os
import time
from typing import Literal
import torch
import tyro
from gsplat import rasterization
# import pycolmap_scene_manager as pycolmap
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # To avoid conflict with cv2
from tqdm import tqdm
from lseg import LSegNet
import json
from pathlib import Path

from utils import (
    load_ply,
    load_checkpoint,
    get_viewmat_from_colmap_image,
    prune_by_gradients,
    prune_by_gradients_json,
    prune_by_gradients_opencv,
    test_proper_pruning_opencv,
    test_proper_pruning,
    test_proper_pruning_json,
    save_gsplat_dict_to_ply,
)


import torch.nn as nn
import sklearn.decomposition
import sklearn
from PIL import Image

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature



def create_feature_field_lseg(splats, batch_size=1, use_cpu=False, inverse_extrinsics=True, resize=[480, 640]):
    device = "cpu" if use_cpu else "cuda"

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(
        torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device)
    )
    net.eval()
    net.to(device)

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    # colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    # K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(colors.shape[0], 512, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    t1 = time.time()

    colors_feats = torch.zeros(
        colors.shape[0], 512, device=colors.device, requires_grad=True
    )
    colors_feats_0 = torch.zeros(
        colors.shape[0], 3, device=colors.device, requires_grad=True
    )
    transformsfile = splats["transform_json_file"]

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"] 
        cy = contents["cy"]
        # if "crop_edge" in contents:
        #     cx -= contents["crop_edge"] 
        #     cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
            # if "crop_edge" in contents:
            #     width -= 2*contents["crop_edge"]
            #     height -= 2*contents["crop_edge"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2 
        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), desc="Feature backprojection (frames)", total=len(frames)):
            # NeRF 'transform_matrix' is a camera-to-world transform


            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = w2c[:3,:3]  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # image_path = os.path.join(image_path, f'{cam_name + extension}')
            # image_name = Path(cam_name).stem
            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
            # resize_ratio = resize[1] / (640 - 2 * contents["crop_edge"]) if "crop_edge" in contents else 0
            resize_ratio = resize[1] / 640
            fx_resize = focal_len_x * resize_ratio
            
            fy_resize = focal_len_y * resize_ratio
            cx_resize = cx * resize_ratio
            cy_resize = cy * resize_ratio
            


            K = torch.tensor(
                [
                    [fx_resize, 0, cx_resize],
                    [0, fy_resize, cy_resize],
                    [0, 0, 1],
                ]
            ).float()

    # images = sorted(colmap_project.images.values(), key=lambda x: x.name)
    # batch_size = math.ceil(len(images) / batch_count) if batch_count > 0 else 1

    # for batch_start in tqdm(
    #     range(0, len(images), batch_size),
    #     desc="Feature backprojection (batches)",
    # ):
    #     batch = images[batch_start : batch_start + batch_size]
    #     for image in batch:
    #         viewmat = get_viewmat_from_colmap_image(image)
            # width = int(K[0, 2] * 2)
            # height = int(K[1, 2] * 2)
            resize_w = resize[1]
            resize_h = resize[0]
            # print("width, height", width, height)
            # print("contents[crop_edge]", contents["crop_edge"])
            

            with torch.no_grad():
                output, _, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors_all,
                    viewmat[None],
                    K[None],
                    width=resize_w,
                    height=resize_h,
                    sh_degree=3,
                )

                output = torch.nn.functional.interpolate(
                    output.permute(0, 3, 1, 2).to(device),
                    size=(480, 480),
                    mode="bilinear",
                )
                output.to(device)
                feats = net.forward(output)
                feats = torch.nn.functional.normalize(feats, dim=1)
                feats = torch.nn.functional.interpolate(
                    feats, size=(resize_h, resize_w), mode="bilinear"
                )[0]
                feats = feats.permute(1, 2, 0)

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats,
                viewmat[None],
                K[None],
                width=resize_w,
                height=resize_h,
            )

            target = (output_for_grad[0].to(device) * feats).sum()
            target.to(device)
            target.backward()
            colors_feats_copy = colors_feats.grad.clone()
            colors_feats.grad.zero_()

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats_0,
                viewmat[None],
                K[None],
                width=resize_w,
                height=resize_h,
            )

            target_0 = (output_for_grad[0]).sum()
            target_0.to(device)
            target_0.backward()

            gaussian_features += colors_feats_copy
            gaussian_denoms += colors_feats_0.grad[:, 0]
            colors_feats_0.grad.zero_()

            # Clean up unused variables and free GPU memory
            del (
                viewmat,
                meta,
                _,
                output,
                feats,
                output_for_grad,
                colors_feats_copy,
                target,
                target_0,
            )
            torch.cuda.empty_cache()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    gaussian_features[torch.isnan(gaussian_features)] = 0
    t2 = time.time()
    print("Time taken for feature backprojection", t2 - t1)

    feats_rendered_save_path = os.path.join(
        Path(transformsfile).parent, f"3dgsback_render_feats")
    if not os.path.exists(feats_rendered_save_path):
        os.makedirs(feats_rendered_save_path)
    
    frames = contents["test_frames"]
    for idx, frame in tqdm(enumerate(frames), desc="Testing (frames)", total=len(frames)):
        # NeRF 'transform_matrix' is a camera-to-world transform
        frame_name = frame["file_path"].split("/")[-1].split(".")[0]
        c2w = np.array(frame["transform_matrix"])
        # get the world-to-camera transform and set R, T
        # w2c = c2w
        # some dataset save world-to-camera, some camera-to-world, careful!
        w2c = np.linalg.inv(c2w)
        # w2c[1:3] *= -1
        R = w2c[:3,:3]  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # image_path = os.path.join(image_path, f'{cam_name + extension}')
        # image_name = Path(cam_name).stem
        viewmat = torch.eye(4).float()  # .to(device)
        viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
        viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
  
    
        K = torch.tensor(
            [
                [fx_resize, 0, cx_resize],
                [0, fy_resize, cy_resize],
                [0, 0, 1],
            ]
        ).float()

        with torch.no_grad():

            feats_rendered, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                gaussian_features,
                viewmat[None],
                K[None],
                width=resize_w,
                height=resize_h,
            )
            feats_rendered = feats_rendered[0]
            feats_rendered = torch.nn.functional.normalize(feats_rendered, dim=-1)
            print("feats_rendered.shape", feats_rendered.shape)
            feats_rendered_save_path_i = os.path.join(
                feats_rendered_save_path, frame_name + '.pth'
            )
            torch.save(feats_rendered, feats_rendered_save_path_i)
            # debug
            # save pca
            feature_reorder = feats_rendered.permute(2,0,1) # [1,512, H, W]
            feature_vis = feature_visualize_saving(feature_reorder)
            feats_rendered_save_path_i_vis = os.path.join(
                feats_rendered_save_path, frame_name + '_vis.png'
            )
            Image.fromarray((feature_vis.cpu().numpy() * 255).astype(np.uint8)).save(feats_rendered_save_path_i_vis)

            output, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_all,
                viewmat[None],
                K[None],
                width=resize_w,
                height=resize_h,
                sh_degree=3,
            )
            render_rgb = output[0, :, :, :3].cpu().numpy() 
            Image.fromarray((render_rgb * 255).astype(np.uint8)).save(
                os.path.join(
                    feats_rendered_save_path, frame_name + '_render_rgb.png'
                )
            )


    return gaussian_features


def main(
    data_root_path: str = "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/original_data/", # subset
    # val_split: str = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/scannetpp_mini_val.txt",
    # full set "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/",  # colmap path
    ply_root_path: str = "/srv/beegfs-benderdata/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/mcmc_3dgs/",  # checkpoint path, can generate from original 3DGS repo
    results_root_dir: str = "./results/scannet/",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    # resize =  [584, 876],
    feature_field_batch_count: int = 1,  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False,  # Run feature field on CPU
    feature: Literal["lseg", "dino"] = "lseg",  # Feature field type
    # start_idx: int = 0,  # Start index for processing
    # end_idx: int = -1,  # End index for processing
    scene_i: str = "scene0011_00",  # Scene index for processing
    rescale: int = 0,  # Rescale factor for images
):
    # val_split_load = np.loadtxt(val_split, dtype=str)
    # if end_idx == -1:
    #     end_idx = len(val_split_load)
    # val_split_load = val_split_load[start_idx:end_idx]

    # for scene_i in val_split_load:
    print("Processing scene:", scene_i)
    result_i_dir = os.path.join(results_root_dir, scene_i)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(result_i_dir, exist_ok=True)
    ply_path_i = os.path.join(ply_root_path, scene_i, 'ckpts', 'point_cloud_30000.ply')
    data_dir = os.path.join(data_root_path, scene_i)
    splats = load_ply(
        ply_path_i, data_dir, rasterizer=rasterizer, dataset="scannet"
    )
    splats_optimized = prune_by_gradients_opencv(splats)

    test_proper_pruning_opencv(splats, splats_optimized)

    # raise NotImplementedError("Debugging, remove this line to continue")

    splats = splats_optimized
    save_gsplat_dict_to_ply(
        ply_path_i.replace(".ply", "_pruned.ply"),
        splats,
    )

    if feature == "lseg":
        if rescale == 0:
            features_save_dir = f"{result_i_dir}/features_lseg_480_640.pt"
            if not os.path.exists(features_save_dir):
                features = create_feature_field_lseg(
                    splats, feature_field_batch_count, run_feature_field_on_cpu,resize=[480, 640]
                )
                torch.save(features, features_save_dir)
            else:
                features = torch.load(features_save_dir)

            xyz_save_dir = f"{result_i_dir}/xyz_lseg_480_640.npy"
            if not os.path.exists(xyz_save_dir):
                xyz = splats_optimized['means'].cpu().numpy()
                xyz = xyz.reshape(-1, 3)
                np.save(xyz_save_dir, xyz)

                
        # except Exception as e:
        #     del features
        #     torch.cuda.empty_cache()
        #     print(f"Error in LSeg feature extraction: 584 876", e)

        # if rescale == 1:
        #     features = create_feature_field_lseg(
        #         splats, feature_field_batch_count, run_feature_field_on_cpu,resize=[320, 480]
        #     )
        #     torch.save(features, f"{result_i_dir}/features_lseg_320_480.pt")
   

        del splats
        del features
        torch.cuda.empty_cache()

            
        # elif feature == "dino":
        #     features = create_feature_field_dino(splats)
        #     print("Features.shape", features.shape)
        #     torch.save(features, f"{result_i_dir}/features_dino.pt")
        # else:
        #     raise ValueError("Invalid field type")

import argparse

def get_arguments():
    argparser = argparse.ArgumentParser(description="Feature Field Extraction")
    argparser.add_argument(
        "--scene_name",
        type=str,
        default="scene0011_00",
        help="Path to the validation split file",
    )
    argparser.add_argument(
        "--rescale",
        type=int,
        default=0,
        help="Rescale factor for images",
    )
    return argparser.parse_args()



if __name__ == "__main__":
    # tyro.cli(main)
    args = get_arguments()
    main(
        scene_i=args.scene_name,
        rescale=args.rescale,
    )

