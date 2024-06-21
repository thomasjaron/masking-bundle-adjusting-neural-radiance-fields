## Better Barf :vomiting_face::heavy_plus_sign::heavy_plus_sign:: Bundle-Adjusting Neural Radiance Fields with Occlusion Detection
(Add authors Thomas, Oliver, Simon)
--------------------------------------

### Prerequisites

This code is developed with Python3 (`python3`). PyTorch 1.9+ is required.  
It is recommended use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Install the dependencies and activate the environment `barf-env` with
```bash
conda env create --file requirements.yaml python=3
conda activate better-barf
```

--------------------------------------
### Running the code

- #### Planar image alignment experiment
  If you want to run the planar image alignment experiment, run:
  ```bash
  python3 train.py --group=<GROUP> --model=planar --yaml=planar --name=<NAME> --seed=3 --barf_c2f=[0,0.4]
  ```
  - Full positional encoding: omit the `--barf_c2f` argument.
  - No positional encoding: add `--arch.posenc!`.

  A video `vis.mp4` will also be created to visualize the optimization process.

- #### Visualizing the results
  We have adjusted the code to visualize the training over TensorBoard and Visdom provided by BARF.
  The TensorBoard events include the following:
  - **SCALARS**: the rendering losses and PSNR over the course of optimization. For BARF, the rotational/translational errors with respect to the given poses are also computed.
  - **IMAGES**: visualization of the RGB images and the RGB/depth rendering.

--------------------------------------
### Codebase structure

The main engine and network architecture can be found in `model/planar.py`.
  
Some tips on using and understanding the codebase:
- The computation graph for forward/backprop is stored in `var` throughout the codebase.
- The losses are stored in `loss`. To add a new loss function, just implement it in `compute_loss()` and add its weight to `opt.loss_weight.<name>`. It will automatically be added to the overall loss and logged to Tensorboard.
- If you are using a multi-GPU machine, you can add `--gpu=<gpu_number>` to specify which GPU to use. Multi-GPU training/evaluation is currently not supported.
  
--------------------------------------

The codebase is based on
```
@inproceedings{lin2021barf,
  title={BARF: Bundle-Adjusting Neural Radiance Fields},
  author={Lin, Chen-Hsuan and Ma, Wei-Chiu and Torralba, Antonio and Lucey, Simon},
  booktitle={IEEE International Conference on Computer Vision ({ICCV})},
  year={2021}
}
```

