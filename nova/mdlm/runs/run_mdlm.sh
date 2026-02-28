#!/bin/bash
# Run from nova/mdlm/ directory: bash runs/run_mdlm.sh

source ../../.venv/mdlm/bin/activate

python train.py \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000
