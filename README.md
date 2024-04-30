# MMGCL

This repository is the official implementation of MMGCL.

MMGCL: Meta Knowledge-Enhanced Multi-view Graph Contrastive Learning for Recommendations

## Requirements

Environment: 16GB(RAM), Ubuntu 16.04 (OS), TITAN RTX (GPU), Xeon Gold 5120 (CPU)

extra packages:

```bash
pip install hyperopt networkx pandas scikit-learn
```

## Getting started

We recommend using docker:
docker image: `docker.io/tensorflow/tensorflow:2.2.0-gpu-jupyter`


```bash
# pull image
docker pull docker.io/tensorflow/tensorflow:2.2.0-gpu-jupyter
# start 
docker run -it --gpus all -v $(pwd):/workspace  -p 8888:8888 --name GMGCL --shm-size 32g docker.io/tensorflow/tensorflow:2.2.0-gpu-jupyter /bin/bash
# install some required packages
pip install hyperopt networkx pandas scikit-learn
# start jupyter
nohup jupyter lab --port 8888 --allow-root --ip 0.0.0.0 2>&1 &
# start and run ./src/GMGCL.ipynb
```

## Data

We release the processed dataset in ``data/datasets``. The processing script of the original data can be obtained from ``dataset.py``.

Due to regulatory restrictions, the KuaiRand dataset can be obtained from https://kuairand.com/.

## Reference

If you use this code as part of your research, please cite the following paper:

```

```

## Reference code

- [Recommenders](https://github.com/microsoft/recommenders)
- [Multi-Graph Graph Attention Network](https://github.com/zuirod/mg-gat)
