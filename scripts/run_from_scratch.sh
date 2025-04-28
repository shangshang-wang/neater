#!/bin/bash


MAMBA_ENV="neater"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

python ./scripts/open_source/neat_slime_rl_flax.py # NEAT for Slime using Flax
#python ./scripts/open_source/backpropneat_nonlinear.py # BackpropNEAT for Circle, Spiral, and XOR using Jax
#python ./scripts/open_source/backpropneat_classification_flax.py # BackpropNEAT for Iris and MNIST using Flax

echo "END TIME: $(date)"
echo "DONE"