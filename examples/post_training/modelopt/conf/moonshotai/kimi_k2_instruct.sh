#!/bin/bash

HF_MODEL_CKPT=/workspace/scratch/moonshotai/Kimi-K2-Instruct
TP=1
ETP=1
EP=1 # Adjust EP to 1 when using PP to distribute layers across more GPUs.
PP=8 # Introduce Pipeline Parallelism to split the 61 layers across 8 GPUs.

