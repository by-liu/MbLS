# MbLS : The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration

[Bingyuan Liu](https://by-liu.github.io/), [Ismail Ben Ayed](https://profs.etsmtl.ca/ibenayed/), Adrian Galdran, [Jose Dolz](https://josedolz.github.io/)

[[`arXiv`](https://arxiv.org/abs/2111.15430)] [[`BibTeX`](#CitingMbLS)]

<div align="center">
  <img src="https://by-liu.github.io/publication/margin-based-label-smoothing/featured_hu1bcfdecf7483f74849c0f7f247a58b3e_176048_720x0_resize_q75_lanczos.jpg" width="100%" height="100%"/>
</div><br/>


## Requirments:

Here is some core dependences:
```
torch==1.7.1+cu110
torchvision==0.8.2+cu110
albumentations==1.0.0
hydra-core==1.1.0
```

For detailed dependences, please check [requirements.txt](requirements.txt). (It maybe a little redundant. We will make it clean soon.)


## Install:

```
python setup.py develop
```


## Data preparation

For CIFAR-10, our code can automatically download the data samples. For the others (Tiny-Imagenet, CUB-200 and VOC 2012), please refer to the official cites for downloading the datasets.

**Important Note** : Before you run the code, please add the absolute path of the data directory for the related data configs in [configs/data](configs/data/).

## Usage:


### Command arguments :

<details><summary>python tools/train_net.py --help</summary>
<p>

```python
train_net is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

data: cifar10, cub, newgroups, tiny_imagenet, voc
loss: ce, focal, focal_adaptive, logit_margin, ls, mmce, penalty_ent
model: deeplabv3, global_pooling, resnet101, resnet101_cifar, resnet101_tiny, resnet110, resnet110_cifar, resnet34, resnet34_cifar, resnet34_tiny, resnet50, resnet50_cifar, resnet50_tiny
optim: adam, sgd
scheduler: multi_step, plateau, step
wandb: my


== Config ==
Override anything in the config (foo.bar=value)

data:
  name: cifar10
  data_root: DATA_PATH
  batch_size: 128
  object:
    trainval:
      _target_: calibrate.data.cifar10.get_train_valid_loader
      batch_size: ${data.batch_size}
      augment: true
      random_seed: ${seed}
      shuffle: true
      num_workers: 4
      pin_memory: true
      data_dir: ${data.data_root}
    test:
      _target_: calibrate.data.cifar10.get_test_loader
      batch_size: ${data.batch_size}
      shuffle: false
      num_workers: 4
      pin_memory: true
      data_dir: ${data.data_root}
model:
  name: resnet50
  num_classes: 10
  pretrained: true
  has_dropout: false
  object:
    _target_: calibrate.net.resnet.build_resnet
    encoder_name: ${model.name}
    num_classes: ${model.num_classes}
    pretrained: ${model.pretrained}
    has_dropout: ${model.has_dropout}
loss:
  name: ce
  ignore_index: -100
  object:
    _target_: torch.nn.CrossEntropyLoss
    ignore_index: ${loss.ignore_index}
    reduction: mean
optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: false
  object:
    _target_: torch.optim.SGD
    lr: ${optim.lr}
    momentum: ${optim.momentum}
    weight_decay: ${optim.weight_decay}
    nesterov: ${optim.nesterov}
scheduler:
  name: multi_step
  milestones:
  - 150
  - 250
  gamma: 0.1
  object:
    _target_: torch.optim.lr_scheduler.MultiStepLR
    milestones: ${scheduler.milestones}
    gamma: ${scheduler.gamma}
    verbose: true
wandb:
  enable: false
  project: ''
  entity: ''
  tags: ''
task: cv
device: cuda:0
seed: 1
log_period: 10
train:
  clip_grad_norm: true
  max_epoch: 200
  resume: false
  keep_checkpoint_num: 1
  keep_checkpoint_interval: 0
calibrate:
  num_bins: 15
  visualize: false
test:
  checkpoint: ''
  save_logits: false


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
</p>
</details>


### Example 

Ours : 
```python
python tools/train_net.py \
    log_period=100 \
    data=tiny_imagenet \
    model=resnet50_tiny model.num_classes=200 \
    loss=logit_margin loss.margin=10.0 loss.alpha=0.1 \
    optim=sgd optim.lr=0.1 optim.momentum=0.9 \
    scheduler=multi_step scheduler.milestones="[40, 60]" \
    train.max_epoch=100
```

Cross entropy (CE) : 
```python
python tools/train_net.py \
    log_period=100 \
    data=tiny_imagenet \
    model=resnet50_tiny model.num_classes=200 \
    loss=ce \
    optim=sgd optim.lr=0.1 optim.momentum=0.9 \
    scheduler=multi_step scheduler.milestones="[40, 60]" \
    train.max_epoch=100
```

Label smoothing (LS) : 
```python
python tools/train_net.py \
    log_period=100 \
    data=tiny_imagenet \
    model=resnet50_tiny model.num_classes=200 \
    loss=ls \
    optim=sgd optim.lr=0.1 optim.momentum=0.9 \
    scheduler=multi_step scheduler.milestones="[40, 60]" \
    train.max_epoch=100
```


## Support for extension or follow-up works

Besides the implementation of our paper, this library could support follow-up works on model calibration with the following features:

* Calibration evaluator, e.g. ECE and AECE
* Reliability diagram
* Experiments tracking with [wandb](https://wandb.ai/)

**More instructions will come soon. Please stay tuned! Thank you.**


## <a name="CitingMbLS"></a>Citing MbLS


```BibTeX
@article{liu2021devil,
  title={The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration}, 
  author={Bingyuan Liu and Ismail Ben Ayed and Adrian Galdran and Jose Dolz},
  journal={arXiv preprint arXiv:2111.15430}
  year={2021}
}
```
