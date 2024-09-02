## MARF :performing_arts:: Masking Bundle-Adjusting Neural Radiance Fields
Authors:

Thomas Jaron-Strugala, 
jaron-strugala@campus.tu-berlin.de
Technical University of Berlin

Oliver Jan Jarosik,
jarosik@campus.tu-berlin.de
Technical University of Berlin

Simon Bonaventura Ertlmaier,
s.ertlmaier@campus.tu-berlin.de
Technical University of Berlin
--------------------------------------

### Prerequisites

This code is developed with Python3 (`python3`). PyTorch 1.9+ is required.  
It is recommended use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Install the dependencies and activate the environment `marf-env` with
```bash
conda env create --file requirements.yaml
conda activate marf
```

--------------------------------------
### Running the code
  If you want to run the planar image alignment, run:
  ```bash
  python3 train.py --group=<GROUP> --model=planar --yaml=planar --name=<NAME> --seed=3 --barf_c2f=[0,0.4]
  ```
  - Full positional encoding: omit the `--barf_c2f` argument.
  - No positional encoding: add `--arch.posenc!`.

  A video `vis.mp4` will also be created to visualize the optimization process.

- #### Visualizing the results
  We have adjusted the code to visualize the training over TensorBoard provided by BARF.
  The TensorBoard events include the following:
  - **SCALARS**: the rendering losses and PSNR over the course of optimization. For BARF, the rotational/translational errors with respect to the given poses are also computed.
  - **IMAGES**: visualization of the RGB images and the RGB rendering.

--------------------------------------
### Codebase structure

The main engine and network architecture can be found in `model/planar.py`.
The codebase is based on
```
@inproceedings{lin2021barf,
  title={BARF: Bundle-Adjusting Neural Radiance Fields},
  author={Lin, Chen-Hsuan and Ma, Wei-Chiu and Torralba, Antonio and Lucey, Simon},
  booktitle={IEEE International Conference on Computer Vision ({ICCV})},
  year={2021}
}
```

and 

```
@misc{chen2022hallucinatedneuralradiancefields,
      title={Hallucinated Neural Radiance Fields in the Wild}, 
      author={Xingyu Chen and Qi Zhang and Xiaoyu Li and Yue Chen and Ying Feng and Xuan Wang and Jue Wang},
      year={2022},
      eprint={2111.15246},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.15246}, 
}
```

