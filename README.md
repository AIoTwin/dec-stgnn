# Simulator for Decentralized Training of Spatio-Temporal Graph Neural Networks
[![issues](https://img.shields.io/github/issues/AIoTwin/dec-stgnn)](https://github.com/AIoTwin/dec-stgnn/issues)
[![forks](https://img.shields.io/github/forks/AIoTwin/dec-stgnn)](https://github.com/AIoTwin/dec-stgnn/network/members)
[![stars](https://img.shields.io/github/stars/AIoTwin/dec-stgnn)](https://github.com/AIoTwin/dec-stgnn/stargazers)
[![License](https://img.shields.io/github/license/AIoTwin/dec-stgnn)](./LICENSE)

## About
This simulator enables experimentation with decentralized training approaches (centralized, federated, server-free, and gossip-based) for Spatio-Temporal Graph Neural Networks in traffic prediction tasks.

## Paper
https://arxiv.org/abs/2412.03188

## Citation
```
@article{kralj2024semi,
  title={Semi-decentralized Training of Spatio-Temporal Graph Neural Networks for Traffic Prediction},
  author={Kralj, Ivan and Giaretta, Lodovico and Ježić, Gordan and Žarko, Ivana Podnar and Girdzijauskas, Šarūnas},
  journal={arXiv preprint arXiv:2412.03188},
  year={2024}
}
```

## Related code
1. STGCN: https://github.com/hazdzz/stgcn

## Dataset
### Source
1. METR-LA: [DCRNN author's Google Drive](https://drive.google.com/file/d/1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC/view?usp=sharing)
2. PEMS-BAY: [DCRNN author's Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=sharing)

## Prerequisites
1. Install micromamba
```
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
```
2. Create micromamba environment
```
micromamba create -n "ENVIRONMENT_NAME" --file environment.yml --channel-priority flexible
```
3. Install PyTorch Geometric
```
pip install torch_geometric
```
4. Install Ray
```
pip install -U "ray[default]"
```

## Configuration

### Hyperparameters
To change hyperparameters configuration, you'll need to go into the main .py file of a scenario you want to simulate and change hyperparameters there. Each scenario can have different hyperparameters that can be changed. Here's a sample:
```
"enable_cuda": True
"seed": 42
"dataset": "pems-bay"
"n_his": 12
"n_pred": 3
"Kt": 3
"stblock_num": 2
"Ks": 3
"graph_conv_type": "cheb_graph_conv"
"enable_bias: True
"droprate": 0.5
"lr": 0.0001
"weight_decay_rate": 0.00001
"batch_size": 32
"epochs": 40
"step_size": 5
"gamma": 0.7
"cloudlet_num": 7
"cloudlet_location_data": "experiment_1"
```

### Cloudlet location data
In order to run experiments while using geographical sensor locations, you'll need to define cloudlet locations as well. The format is in .json, and you need to store them in ```"/locations/DATASET_NAME/locations.json"``` file.
Example:
```
{
    "experiment_1":
    {
        "cloudlets": {
            "cloudlet_1": {"id": 0, "lat": 37.395, "lon": -122.057, "color": "blue"},
            "cloudlet_2": {"id": 1, "lat": 37.335, "lon": -122.040, "color": "gray"},
            "cloudlet_3": {"id": 2, "lat": 37.400, "lon": -121.975, "color": "red"},
            "cloudlet_4": {"id": 3, "lat": 37.390, "lon": -121.915, "color": "green"},
            "cloudlet_5": {"id": 4, "lat": 37.345, "lon": -121.870, "color": "purple"},
            "cloudlet_6": {"id": 5, "lat": 37.290, "lon": -121.890, "color": "cyan"},
            "cloudlet_7": {"id": 6, "lat": 37.285, "lon": -121.977, "color": "orange"}
        },
        "radius_km": 8
    }
}
```

## Usage
Make sure you're in your create micromamba environment
```
micromamba activate "ENVIRONMENT_NAME"
```

Run any .py file that has main function in order to run a simulation for that specific scenario.

Example:
```
python 2d-matrix-ray-actors-semi-dec-fl-architecture-distance.py
```