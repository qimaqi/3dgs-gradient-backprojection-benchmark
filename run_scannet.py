import os 
import sys
import numpy as np

# validation_dir =
# validation_set = np.loadtxt(validation_dir, dtype=str)

import time



def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Run 3DGS Gradient Backprojection Benchmark')
    parser.add_argument('--split', type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/scannet_mini_val.txt', help='Split name')
    parser.add_argument('--rescale', type=int, default=0, help='rescale custom')
    parser.add_argument('--start_idx',type=int, default=0, help='start idx')
    parser.add_argument('--end_idx',type=int, default=-1, help='end idx')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_arguments()
    print(args)
    split = args.split
    rescale = args.rescale

    validation_set = np.loadtxt(args.split, dtype=str)
    validation_set = sorted(validation_set)
    if args.end_idx == -1:
        args.end_idx = len(validation_set)
    validation_set = validation_set[args.start_idx:args.end_idx]

    # validation_set = ['scene0329_01', 'scene0435_01']
    start_time = time.time()
    for scene in validation_set:
        # scene = scene.split('/')[-1]
        # if args.rescale == 0:
        #     potential_results_dir = os.path.join('/insait/qimaqi/workspace/3dgs-gradient-backprojection-benchmark/results/scannet', scene, 'features_lseg_480_640.pt')
        # elif args.rescale == 1:
        #     potential_results_dir = os.path.join('/insait/qimaqi/workspace/3dgs-gradient-backprojection-benchmark/results/scannet',scene, 'features_lseg_320_480.pt')

        # if os.path.exists(potential_results_dir):
        #     print("Found potential results dir: ", potential_results_dir)
        #     continue   
        print(scene)
        os.system("python backproject_ply_scannet.py --scene_name {} --rescale {}".format(scene, rescale))
    
    end_time = time.time()
    print("Total time: ", end_time - start_time)