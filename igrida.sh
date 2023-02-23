#!/bin/bash

oarsub \
    -O igrida.o \
    -E igrida.e \
    -l /gpu_device=1/core=4,walltime=01:00:00 \
    "bash run.sh"
