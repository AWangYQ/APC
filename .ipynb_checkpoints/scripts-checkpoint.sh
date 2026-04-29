#!/bin/bash

# single domain
# MSMT17
python train.py --config_file configs/MSMT17/APC.yml

# Market1501
python train.py --config_file configs/Market1501/APC.yml

# Duke
python train.py --config_file configs/Duke/APC.yml

# Domain Generation

python train.py --config_file configs/MSMT17/APC_MS2D.yml # MSMT17 2 Duke

python train.py --config_file configs/MSMT17/APC_MS2MA.yml # MSMT17 2 Market1501

python train.py --config_file configs/MSMT17/APC_multi.yml # Multis 2 MSMT17

python train.py --config_file configs/Market1501/APC_MA2D.yml # Market1501 2 Duke

python train.py --config_file configs/Market1501/APC_MA2MS.yml # Market1501 2 MSMT17

python train.py --config_file configs/Market1501/APC_multi.yml # Multis 2 Market1501

python train.py --config_file configs/Duke/APC_D2MA.yml  # Duke 2 Market1501

python train.py --config_file configs/Duke/APC_D2MS.yml # Duke 2 MSMT17

python train.py --config_file configs/Duke/APC_multi.yml # Multis 2 Duke


