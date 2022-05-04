# VolSDF-reproduce

This repo reproduces [VolSDF](https://arxiv.org/abs/2106.12052), and its official implementation is [here](https://github.com/lioryariv/volsdf).

## TODO

- [ ] Implement background rendering.
- [ ] Optimize training process to avoid OOM and improve the utilization of graphics memory.

## Dataset 

It's only support DTU dataset now.
Please refer to the procedure in official github.

## Usage

### train

```shell

python train.py --config ./config/DTU.txt \{--pretrained_model CKPT\}

```

`--pretrained_model` is used to resume training.

### test

```shell

python train.py --config ./config/DTU.txt \{--pretrained_model CKPT\} --scan_id SCAN_ID --render_only

```

## Citation

```
@inproceedings{yariv2021volume,
  title={Volume rendering of neural implicit surfaces},
  author={Yariv, Lior and Gu, Jiatao and Kasten, Yoni and Lipman, Yaron},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
