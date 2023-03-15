#!/bin/bash

oarsub \
    -O igrida.o \
    -E igrida.e \
    -l /gpu_device=1/core=12,walltime=02:00:00 \
    "bash run.sh"
