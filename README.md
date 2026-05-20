# CSO: Improving the Sensitivity of Backdoor Detectors via Class Subspace Orthogonalization


This the official Pytorch implementation of our paper "Improving the Sensitivity of Backdoor Detectors via Class Subspace Orthogonalization", accepted by ICML 2026.

## Requirements

To install the requirements:

```bash
pip install -r requirements.txt
```

## Build Mixed-Label(ML) attacks

Crafts 1-to-1 BadNet (dirty-label and mixed-label, DPR=0.1%) and
trains 10 victims per variant on CIFAR-10:

```bash
./run_attack.sh
```

Outputs land in `attack/attack<i>/` and `attack/model<i>/`
(`model.pth`, `model_ml_0.01_2.pth`).

You can also create ML attacks on other attack types by changing `attack/util.py` and `attack/attack_crafting.py`.

## MMBD-CSO

Single checkpoint:

```bash
python mmbd_cso.py \
    --ckpt_path attack/model0/model.pth
```

Sweep all 20 models (both original and ML attacks):

```bash
./run_mmbd.sh
```


## Pretrained Poisoned Models
TODO
