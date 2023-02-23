#!/bin/bash

cd /nfs/nas4/serpico_1/serpico/qrapilly

module load spack/singularity
singularity --nv -B /srv:/srv MC-net_image.sif

cd MC-Snake_net
python src/train.py