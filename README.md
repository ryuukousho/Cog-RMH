# Cog-RMH: Cognition-based Recalling Multiview History for Event Forecasting in Temporal Knowledge Graph

This is the released codes of the submission to the IEEE Transactions on Knowledge and Data Engineering (TKDE). Please note that no commercial use is permitted until the paper is accepted.

## Environment

```shell
python==3.10.9
torch==2.2.1+cu118
dgl==2.1.0+cu118
tqdm==4.66.2
numpy==1.26.4
```

## Introduction

- ``src``: Python scripts.
- ``results``: Model files that replicate the reported results in our paper.
- ``data``: TKGs used in the experiments.

## Training Command

```shell
cd src
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6
```

```shell
cd src
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9
```

```shell
cd src
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 12
```

```shell
cd src
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9
```

## Testing Command

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6 --test
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9 --test
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 12 --test
```

```shell
CUDA_VISIBLE_DEVICES=1 python main.py --model CogRMH --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9 --test
```

## Acknowledgements

The source codes take [ReTIN](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12186) as the backbone to implement our proposed method. Please cite both our work and [ReTIN](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12186) if you find this repository is helpful for your research.
