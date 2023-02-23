#!/bin/bash

cd /nfs/nas4/serpico_1/serpico/qrapilly/MC-Snake_net

. /etc/profile.d/modules.sh
module load spack/singularity
singularity exec -B /srv:/srv --nv ../MC-net_image.sif python src/train.py
