# Decompressing Knowledge Graph Representations for Link Prediction

This is the code we used in our paper
>[Decompressing Knowledge Graph Representations for Link Prediction](https://arxiv.org/pdf/1911.04053.pdf)

>Xiang Kong\*, Xianyang Chen\*, Eduard Hovy (*: equal contribution)

## Requirements

Python 3 

PyTorch >= 1.0

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install the requirements `pip install -r requirements.txt`
2. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`
3. Run the preprocessing script for WN18RR, FB15k-237: `sh preprocess.sh`
## Running a model


```
CUDA_VISIBLE_DEVICES=0 python main.py model DistMultDecompress dataset FB15k-237 lr  0.001 hidden_drop 0.2 epochs 30 process False

```
will run a  model DistMultDecompress on FB15k-237.
More models are listed in model.py



## Acknowledgement

This repo is adapted from ConvE (https://github.com/TimDettmers/ConvE). Thanks!

## Citation

If you found this codebase or our work useful please cite us:
```
@article{kong2019decompressing,
  title={Decompressing Knowledge Graph Representations for Link Prediction},
  author={Kong, Xiang and Chen, Xianyang and Hovy, Eduard},
  journal={arXiv preprint arXiv:1911.04053},
  year={2019}
}
```
