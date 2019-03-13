#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 --master_port 2000 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}