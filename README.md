# private_flocking
Code accompanying the paper
> ["An Adversarial Approach to Private Flocking in Mobile Robot Teams"](https://arxiv.org/abs/1909.10387)\
> in _IEEE Robotics and Automation Letters_, vol. 5, no. 2, pp. 1009-1016, April 2020\
> _doi: 10.1109/LRA.2020.2967331_

## Requirements
Simulation environment
```
Unreal Engine (UE4) v4.18
AirSim
```
Learning libraries
```
Python >= 3.5
PyGMO == 2.10
PyTorch == 1.2.0
torchvision == 0.4.0
```
Instructions to compile and run AirSim with UE4can be found [here](https://github.com/JacopoPan/a-minimalist-guide/blob/master/Part3-Using-AirSim.md#building-unreal-engine-4-and-airsim-from-source)

Then
```
$ git clone https://github.com/proroklab/private_flocking.git
$ cd private_flocking/
```

## Discriminator pre-training
Before the co-optimization, the discriminator needed to be pre-trained.\
This step can be skipped by using our [pretrain-weights.py](https://github.com/proroklab/private_flocking/blob/master/pretrain-weights.pt).

**Collect pre-training data**
```
$ cd pretrain_discriminator && python 1_collect_cnn_data.py
```

**Pre-training**
```
$ cd pretrain_discriminator/scripts
$ ./2_run_parse_cnn_data.sh
$ ./3_run_pretrain.sh
$ ./4_tensorboard.sh
```
* After pre-training, weights.pt need to be manually copied into folder private_flocking and renamed as pretrain-weights.py

## Co-optimization
```
$ cd scripts && ./run.sh
```
Please note that arguments used for the co-optimization experiment are stored in [config.py](https://github.com/proroklab/private_flocking/blob/master/config.py)

## Visualization
Visualize a single flocking simulation.
```
$ cd scripts && ./summary-plot.sh
```
Visualize and compare up to 4 flocking simulations.
```
$ cd scripts && ./summary-plot.sh
```
Visualize the co-optmization process.
```
$ cd scripts && ./evolution-plot.sh
```
where the optimization id and simulation id for visualization should be changed in each script.

## Citation
If you like, please cite our paper as:
```
@ARTICLE{zheng2020adversarial,
  author={Zheng, Hehui and Panerati, Jacopo and Beltrame, Giovanni and Prorok, Amanda},
  journal={IEEE Robotics and Automation Letters}, 
  title={An Adversarial Approach to Private Flocking in Mobile Robot Teams}, 
  year={2020},
  volume={5},
  number={2},
  pages={1009-1016},
  doi={10.1109/LRA.2020.2967331}}
```
