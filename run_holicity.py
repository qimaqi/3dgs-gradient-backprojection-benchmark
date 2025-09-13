import os 
import sys
import numpy as np

# validation_dir =
# validation_set = np.loadtxt(validation_dir, dtype=str)

import time



def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Run 3DGS Gradient Backprojection Benchmark')
    parser.add_argument('--split', type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/matterport3d_mini_test.txt', help='Split name')
    parser.add_argument('--rescale', type=int, default=0, help='rescale custom')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_arguments()
    print(args)
    split = args.split
    rescale = args.rescale

    validation_set = np.loadtxt(args.split, dtype=str)
    validation_set = sorted(validation_set)
    start = time.time()
    for scene in validation_set:
        # scene = scene.split('/')[-1]
        if args.rescale == 0:
            potential_results_dir = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/holicity/', scene, 'features_lseg_512_512.pt')
        # elif args.rescale == 1:
        #     potential_results_dir = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/scannetpp',scene, 'features_lseg_480_640.pt')
        # elif args.rescale == 2:
        #     potential_results_dir = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/scannetpp', scene, 'features_lseg_240_320.pt')

        if os.path.exists(potential_results_dir):
            print("Found potential results dir: ", potential_results_dir)
            continue   
        print(scene)
        os.system("python backproject_ply_holicity.py --scene_name {} --rescale {}".format(scene, rescale))
    end = time.time()
    print("Total time taken: ", end - start)
