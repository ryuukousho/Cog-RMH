# Cog-RMH: Cognition-based Recalling Multiview History for Event Forecasting in Temporal Knowledge Graph

This is the released codes of the submission to the ACM Transactions on Information Systems (TOIS):

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
python main.py --model CogRMH --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6
```

```shell
cd src
python main.py --model CogRMH --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9
```

```shell
cd src
python main.py --model CogRMH --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 12
```

```shell
cd src
python main.py --model CogRMH --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9
```

## Testing Command

```shell
python main.py --model CogRMH --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6 --test
```

```shell
python main.py --model CogRMH --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9 --test
```

```shell
python main.py --model CogRMH --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 12 --test
```

```shell
python main.py --model CogRMH --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9 --test
```
