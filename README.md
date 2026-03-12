# MARL-HeteroUAVs

Code for the project: The dynamic collaborative perception and path planning of Heterogeneous UAVs  under telecommunication limitations based on MARL.  

This is an ongoing research project. Code and experimental results are **subject to change**.

## Current progress

- We have established a 2.5D heterogeneous UAV simulation environment based on the second-order integral **dynamic physics engine**.

- We have respectively implemented the **MAPPO** and the **MASAC** algorithm, both of which incorporated the **multi-head attention** mechanism.



# Project structure

```
MARL-HeteroUAVSwarm/
├── algorithms/
│   ├── __init__.py
│   ├── buffer.py
│   ├── mappo.py
│   ├── masac.py
│   └── modules.py
├── results/
├── rl_env/
│   ├── __init__.py
│   ├── config.py
│   ├── core.py
│   ├── cover_scan.py
│   ├── test_env.py
│   └── scenarios/
│       ├── __init__.py
│       └── uav_mission.py
├── scripts/
│   └── __init__.py
├── README.md
└── requirements.txt
```



## Installation

First, create a conda environment using Python=3.9

```
conda create -n marl_uav python=3.9
conda activate marl_uav
```

Then, install the dependencies

```
pip install -r requirements
```

