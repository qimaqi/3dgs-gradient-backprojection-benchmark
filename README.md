# Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting

This repository contains the code for the paper **Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting**.


Project page: https://jojijoseph.github.io/3dgs-backprojection
Preprint: https://arxiv.org/abs/2411.15193


## Setup

Please install the dependencies listed in `requirements.txt` via `pip install -r requirements.txt`. Download `lseg_minimal_e200.ckpt` from https://mitprod-my.sharepoint.com/:u:/g/personal/jkrishna_mit_edu/EVlP4Ggf3OlMgDACHNVYuIYBZ4JNi5nJCQA1kXM-_nrB3w?e=XnPT39 and place it in the `./checkpoints` folder. 

Other than that, it's a self-contained repo. Please feel free to raise an issue if you face any problems while running the code.

## Demo



https://github.com/user-attachments/assets/1aecd2d1-8e16-499e-98ce-a1667be5114d

Left: Original rendering, Mid: Extraction, Right: Deletion

Sample data (garden) can be found [here](https://drive.google.com/file/d/1cEPby9zWgG40dJ4eRiHu15Jdg7FgvTdG/view?usp=sharing). Please create a folder named `data` on root folder and extract the contents of zip file to that folder.

To backproject the features run 

```bash
python transfer.py --help
```

Once backprojection is completed, run the following to see the segmenation results.

```bash
python segment.py --help
```


Trained Mip-NeRF 360 Gaussian splat models (using [gsplat](https://github.com/nerfstudio-project/gsplat) with data factor = 4) can be found [here](https://drive.google.com/file/d/1ZCTgAE6vZOeUBdR3qPXdSPY01QQBHxeO/view?usp=sharing). Extract them to `data` folder.

## Acknowledgements

A big thanks to the following tools/libraries, which were instrumental in this project:

- [gsplat](https://github.com/nerfstudio-project/gsplat): 3DGS rasterizer.
- [LSeg](https://github.com/isl-org/lang-seg) and [LSeg Minimal](https://github.com/krrish94/lseg-minimal) : To generate features to be backprojected.


## Citation
If you find this paper or the code helpful for your work, please consider citing our work,
```
@misc{joseph2024gradientweightedfeaturebackprojectionfast,
      title={Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting}, 
      author={Joji Joseph and Bharadwaj Amrutur and Shalabh Bhatnagar},
      year={2024},
      eprint={2411.15193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15193}, 
}
```
