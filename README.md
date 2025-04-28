# Neater

Practing project to rerpoduce NEAT and BackpropNEAT in Jax and Flax.

## Quickstart

Env setup:

```bash
conda update -n base -c defaults conda -y
conda install -n base -c conda-forge mamba -y
mamba shell init --shell bash --root-prefix=~/.local/share/mamba

mamba create -n neater python=3.11 -y && mamba activate neater
pip install -r requirements.txt && mamba deactivate
```

Run the NEAT and BackpropNEAT experiments based on open source codebases:

```bash
./scripts/run_open_source.sh
```

Run the NEAT and BackpropNEAT experiments built from scratch:

```bash
./scripts/run_from_scratch.sh
```
