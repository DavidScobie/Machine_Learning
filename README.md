# UCL Module MPHY0041 Machine Learning in Medical Imaging

This is the repository for the part: **Deep Learning for Medical Imaging**  

**Lecturer:** Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Development environments 
There is no requirement, in tutorials or assessed coursework, for what the development environment that needs to be used. However, technical support from this module is available for the setups detailed in the following documents, under `docs` folder.

### Python environment
The tutorials require a few dependencies, numpy, scipy, matplotlib. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below).

Miniconda or Anaconda is recommended to set up the Python development environment. After you [install Miniconda or Anaconda], create a module environment:
```bash
conda create --name mphy0041 numpy scipy matplotlib h5py tensorflow pytorch torchvision
```
Then activate the environment:
```bash
conda activate mphy0041
```
To return to conda base environment:
```bash
conda deactivate
```

### Optional - Install TensorFlow or PyTorch in `mphy0041` with GPU support
For TensorFlow users, 
```bash
conda install tensorflow-gpu -c anaconda 
```

For PyTorch users,
```bash
conda install pytorch cudatoolkit=11.0 -c pytorch
```

### Cheat - Use TensorFlow and PyTorch on Google Colab
[Google Colab](https://colab.research.google.com/)