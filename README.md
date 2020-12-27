# Deep Learning for Medical Imaging

**Part of the UCL Module MPHY0041: Machine Learning in Medical Imaging**

**Lecturer:** Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Development environments

### Python and conda environment
The tutorials require a few dependencies, numpy, scipy, matplotlib, in addition to one of the two deep learning frameworks. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). Conda is recommended to set up the Python development environment with required dependencies. 

There is no requirement, in tutorials or assessed coursework, for what the development environment that needs to be used. However, technical support in this module is available for the setups detailed in [Supported Development Environment](docs/env.md). 

### Deep learning frameworks
Module tutorials are supported by both [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/). 

Learning materials for TensorFlow for Medical Imaging are recommended in [Learning TensorFlow for Medical Imaging](docs/tensorflow.md).

Learning materials for PyTorch for Medical Imaging are recommended in [Learning PyTorch for Medical Imaging](docs/pytorch.md).


## Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First, set up the environment:
``` bash
conda create --name mphy0041 numpy scipy matplotlib h5py tensorflow pytorch
conda activate mphy0041
```

Then, go to each individual tutorial and run the `data.py` script to download tutorial data, after replacing `tutorial_subfolder_name`:
``` bash
cd tutorial_subfolder_name  # e.g. `cd segmentation`
python data.py
```

Run one of the training scripts:
``` bash
python train_pt.py
```
or 
``` bash
python train_tf2.py
```

Visualise example data and (predicted) labels:
``` bash
python visualise.py
```

### 2D Image clssification
[Anatomical structure classification on ultrasound images](tutorials/classification)

### 3D Image segmentation
[Segmentation of organs on 3D MR images](tutorials/segmentation)

### Image registration*
[Unsupervised registration of CT image slices](tutorials/registration)

### Image synthesis*
[Ultrasound image simulation](tutorials/synthesis)
