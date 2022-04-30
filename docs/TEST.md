# Testing with trained model

* Dataset : TinyImageNet
* Network : ResNet-50

### Running command
```python
python tools/test_net.py \
    data=tiny_imagenet \
    data.data_root=[The Root Directory Of The Dataset] \
    model=resnet50_tiny \
    model.num_classes=200 \
    hydra.run.dir=[The Directory Of The Downloaded Checkpoint] \
    test.checkpoint=[The Filename Of The Checkpoint]
```


### Running Examples

#### CE
```python
python tools/test_net.py \
    data=tiny_imagenet \
    data.data_root=/home/bliu/work/Data/tiny-imagenet-200 \
    model=resnet50_tiny \
    model.num_classes=200 \
    hydra.run.dir=outputs/best_models/tiny_resnet50 \
    test.checkpoint=resnet50_tiny-ce-best.pth
```

```
[2022-04-30 18:26:32,242 INFO][tester.py:123 - log_eval_epoch_info] -
+---------+---------+---------+---------+
| samples | acc     | acc_5   | macc    |
+---------+---------+---------+---------+
| 10000   | 0.65020 | 0.85960 | 0.65020 |
+---------+---------+---------+---------+
[2022-04-30 18:26:32,286 INFO][tester.py:124 - log_eval_epoch_info] -
+---------+---------+---------+---------+---------+
| samples | nll     | ece     | aece    | cece    |
+---------+---------+---------+---------+---------+
| 10000   | 1.40984 | 0.03728 | 0.03687 | 0.00137 |
+---------+---------+---------+---------+---------+
```

### LS
```python
python tools/test_net.py \
    data=tiny_imagenet \
    data.data_root=/home/bliu/work/Data/tiny-imagenet-200 \
    model=resnet50_tiny \
    model.num_classes=200 \
    hydra.run.dir=outputs/best_models/tiny_resnet50 \
    test.checkpoint=resnet50_tiny-ls-best.pth
```

```
[2022-04-30 18:27:34,880 INFO][tester.py:123 - log_eval_epoch_info] -
+---------+---------+---------+---------+
| samples | acc     | acc_5   | macc    |
+---------+---------+---------+---------+
| 10000   | 0.65780 | 0.86190 | 0.65780 |
+---------+---------+---------+---------+
[2022-04-30 18:27:34,880 INFO][tester.py:124 - log_eval_epoch_info] -
+---------+---------+---------+---------+---------+
| samples | nll     | ece     | aece    | cece    |
+---------+---------+---------+---------+---------+
| 10000   | 1.41337 | 0.03165 | 0.03159 | 0.00138 |
+---------+---------+---------+---------+---------+
```


#### MbLS
```python
python tools/test_net.py \
    data=tiny_imagenet \
    data.data_root=/home/bliu/work/Data/tiny-imagenet-200 \
    model=resnet50_tiny \
    model.num_classes=200 \
    hydra.run.dir=outputs/best_models/tiny_resnet50 \
    test.checkpoint=resnet50_tiny-mbls-best.pth
```

```
[2022-04-30 18:23:46,768 INFO][tester.py:123 - log_eval_epoch_info] -
+---------+---------+---------+---------+
| samples | acc     | acc_5   | macc    |
+---------+---------+---------+---------+
| 10000   | 0.64740 | 0.86030 | 0.64740 |
+---------+---------+---------+---------+
[2022-04-30 18:23:46,768 INFO][tester.py:124 - log_eval_epoch_info] -
+---------+---------+---------+---------+---------+
| samples | nll     | ece     | aece    | cece    |
+---------+---------+---------+---------+---------+
| 10000   | 1.43714 | 0.01641 | 0.01730 | 0.00140 |
+---------+---------+---------+---------+---------+
```

