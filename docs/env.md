
# Supported Development Environment

After you [install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/), create a module environment:
```bash
conda create --name mphy0041 numpy scipy matplotlib h5py tensorflow pytorch
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