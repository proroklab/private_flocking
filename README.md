# private_flocking
Code accompanying the paper
> [An Adversarial Approach to Private Flocking in Mobile Robot Teams](https://arxiv.org/abs/1909.10387)\
> Hehui Zheng (1), Jacopo Panerati (2), Giovanni Beltrame (2), Amanda Prorok (1)\
> ((1) University of Cambridge, (2) Polytechnique Montreal)\
> _arXiv: 1909.10387_.

## Requirements
```
Unreal Engine release 4.18 (UE4), AirSim,
Python >= 3.5, PyTorch == 1.2.0, torchvision == 0.4.0, PyGMO == 2.10
```

## Pre-training discriminator
Before the co-optimization, the discriminator needed to be pre-trained.\
This step can be skipped by using our [pretrain-weights.py](https://github.com/proroklab/private_flocking/blob/master/pretrain-weights.pt).

**Collect pre-training data**
```
cd pretrain_discriminator && python 1_collect_cnn_data.py
```

**Pre-training**
```
cd pretrain_discriminator/scripts
./2_run_parse_cnn_data.sh
./3_run_pretrain.sh
./4_tensorboard.sh
```
* After pre-training, weights.pt need to be manually copied into folder private_flocking and renamed as pretrain-weights.py

## Co-optimization
```
cd scripts && ./run.sh
```
Please note that arguments used for the co-optimization experiment are stored in [config.py](https://github.com/proroklab/private_flocking/blob/master/config.py)

## Visualization
Visualize a single flocking simulation.
```
cd scripts && ./summary-plot.sh
```
Visualize and compare up to 4 flocking simulations.
```
cd scripts && ./summary-plot.sh
```
Visualize the co-optmization process.
```
cd scripts && ./evolution-plot.sh
```
where the optimization id and simulation id for visualization should be changed in each script.

## Citation
If you use any part of this code in your research, please cite our paper:
```
updated once on arXiv
```
