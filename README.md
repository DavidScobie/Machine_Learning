# Deep Learning
Part of the UCL Module MPHY0041: Machine Learning in Medical Imaging

**Lecturer:** Yipeng Hu <yipeng.hu@ucl.ac.uk>

|**Other Contacts**   | Email                       | Role        |
|---------------------|-----------------------------|-------------|
|Dr Andre Altmann     | <a.altmann@ucl.ac.uk>       | Module Lead |
|Dr Adria Casamitjana | <a.casamitjana@ucl.ac.uk>   | Tutor       |
|Zac Baum             | <zachary.baum.19@ucl.ac.uk> | Tutor       |


## 1. Development environments

### Python and conda environment
The tutorials require a few dependencies, numpy, matplotlib, in addition to one of the two deep learning frameworks. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). Conda is recommended to set up the Python development environment with required dependencies. 

There is no requirement, in tutorials or assessed coursework, for what the development environment that needs to be used. However, technical support in this module is available for the setups detailed in [Supported Development Environment](docs/env.md). 

### Deep learning frameworks
Module tutorials are implemented in both [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/). 

Learning materials for TensorFlow for Medical Imaging are recommended in [Learning TensorFlow for Medical Imaging](docs/tensorflow.md).

Learning materials for PyTorch for Medical Imaging are recommended in [Learning PyTorch for Medical Imaging](docs/pytorch.md).

### General programming and developping
This is outside of the module scope, but you might find useful materials and links in [the UCL Module MPHY0030 Programming Foundations in Medical Image Analysis](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030), including: 
- [Supported development environment for Python](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030/-/blob/master/docs/dev_env_python.md)
- [Work with Git](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030/-/blob/master/docs/dev_env_git.md)
- [A Roadmap to Learning Python and Machine Learning by Yourself](https://weisslab.cs.ucl.ac.uk/WEISSTeaching/mphy0030/-/blob/master/docs/diy_python_ml.md)


## 2. Tutorials
### Quick start
To run the tutorial examples, follow the instruction below.

First, set up the environment:
``` bash
conda create --name mphy0041 numpy matplotlib h5py tensorflow pytorch
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

### Image clssification
[Anatomical structure classification on 2D ultrasound images](tutorials/classification)

### Image segmentation
[Segmentation of organs on 3D MR images](tutorials/segmentation)

### Image registration*
[Unsupervised registration of CT image slices](tutorials/registration)

### Image synthesis*
[Ultrasound image simulation](tutorials/synthesis)
