#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

if [ -x "../.pixi/envs/default/bin/python" ]; then
    ../.pixi/envs/default/bin/python run_openMM.py > energy.log
else
    pixi run --frozen python run_openMM.py > energy.log
fi
