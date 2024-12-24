## Acknowledgement

This implementation of the [Clip-Adapter paper](https://arxiv.org/pdf/2110.04544) uses the [Dassl PyTorch extension](https://github.com/KaiyangZhou/Dassl.pytorch/tree/master?tab=readme-ov-file), a popular PyTorch extension used to conduct research experiments in the fields of Domain Adaptation and Semi-Supervised Learning. This implementation also takes inspiration from the experiments in the [CoOp paper](https://arxiv.org/abs/2109.01134). Both Dassl and CoOp were points of reference for the authors of Clip-Adapter as well.

## How to Install

Run the following commands from the dassl folder:
```bash
# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

Then navigate to the clip_adapter folder and run the following command:
```bash
# Install clip_adapter specific dependencies
pip install -r requirements.txt
```

Follow [DATASETS.md](clip_adapter/DATASETS.md) to install the datasets.
Follow [CLIP_ADAPTER.md](clip_adapter/CLIP_ADAPTER.md) to run the experiments to reproduce the results of Clip-Adapter.
