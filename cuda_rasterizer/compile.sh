#!/bin/sh

/usr/local/cuda/bin/nvcc --keep-device-functions -O3 -ptx -I/mnt/d/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/include --use_fast_math optix.cu -o optix.ptx
