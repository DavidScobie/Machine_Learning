# UCL Module MPHY0041: Machine Learning in Medical Imaging

**Deep Learning for Medical Imaging**  

**Lecturer:** Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Development environments 
There is no requirement, in tutorials or assessed coursework, for what the development environment that needs to be used. However, technical support from this module is available for the setups detailed under the `docs` folder.

### Python and conda environment
The tutorials require a few dependencies, numpy, scipy, matplotlib, in addition to the deep learning framework  Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). Conda is recommended to set up the Python development environment with required dependencies. 

The supported Python/conda environment is detailed in `docs/env.md` 

### Deep learning frameworks
Module tutorials are supported by both [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/). 


Learning materials for TensorFlow for Medical Imaging are recommended in `docs/tensorflow`.

Learning materials for PyTorch for Medical Imaging are recommended in `docs/pytorch`.


## Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First set up the environment.
``` bash
conda create --name mphy0041 numpy scipy matplotlib h5py tensorflow pytorch
conda activate mphy0041
```

Then go to each individual tutorial and run the `data.py` script to download tutorial data, e.g.:
``` bash
cd classificaion
python data.py
```

Run one of the training scripts.

``` bash
python train_pt.py
```

or 

``` bash
python train_tf2.py
```

### 2D Image clssification

### 3D Image segmentation

### Deep reinforcement learning