#!/bin/bash


MAMBA_ENV="neater"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

python ./scripts/open_source/neat_slime.py # NEAT for Slime based on open source codebases
#python ./scripts/open_source/backpropneat_nonlinear.py # BackpropNEAT for Circle, Spiral, and XOR based on open source codebases

echo "END TIME: $(date)"
echo "DONE"