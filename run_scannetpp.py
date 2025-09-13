import os 
import sys
import numpy as np

# validation_dir =
# validation_set = np.loadtxt(validation_dir, dtype=str)

import time 



def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Run 3DGS Gradient Backprojection Benchmark')
    parser.add_argument('--split', type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/scannetpp_mini_val.txt', help='Split name')
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
    # validation_set_done=['09c1414f1b', '0d2ee665be', '38d58a7a31']
    # validation_set = [scene for scene in validation_set if scene not in validation_set_done]
    # validation_set = ['0d2ee665be']
    for scene in validation_set:
        # scene = scene.split('/')[-1]
        # if args.rescale == 0:
        #     potential_results_dir = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/scannetpp', scene, 'features_lseg_584_876.pt')
        # elif args.rescale == 1:
        #     potential_results_dir = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/scannetpp',scene, 'features_lseg_480_640.pt')
        # elif args.rescale == 2:
        #     potential_results_dir = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/scannetpp', scene, 'features_lseg_240_320.pt')

        # if os.path.exists(potential_results_dir):
        #     print("Found potential results dir: ", potential_results_dir)
        #     continue   
        print(scene)
        os.system("python backproject_ply_scannetpp.py --scene_name {} --rescale {}".format(scene, rescale))
    
    end = time.time()
    print("Total time taken: ", end - start)