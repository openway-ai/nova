# Non-autoregressive Optimization & Vast Architecture (NOVA)

![nanochat logo](dev/nova.png)

This project aims to explore non-autoregressive (NAR) large language models and investigate their potential to scale effectively with increasing training data and model size.
While autoregressive (AR) models have demonstrated strong scaling behavior, non-autoregressive large models remain underexplored, especially in the context of foundation-scale training.
We welcome contributions from the community to push this boundary, and make non-autoregressive models truly powerful and practical.



## Why non-autoregressive models?

AR models are fundamentally limited by their sequential generation process, which restricts parallelism and token-by-token error accumulation.
NAR models offer structural advantages to enhance parallelism, safety and controllability
This year, it is exciting to see more substantial progress in NAR modeling this year, which includes:

- **EBT** Energy-Based Transformers are Scalable Learners and Thinkers ICLR as an ORAL paper [Openreview](https://openreview.net/forum?id=ZBj3Qp1bYg)

- **EBM** Logical Inteligence, the first startup company that focuses on developing energy-based foundation models. See [Logical Inteligence](https://logicalintelligence.com/).

- **MDLM** The first product-level diffusion LM has been online. See [Mercury-2](https://www.inceptionlabs.ai/blog/introducing-mercury-2).

- We are looking forward to more exciting research and product in the future.


## What we provide?

With this repository, you are ready to train NAR models on a 100B pretraining dataset, and explore the scaling laws of NAR models. The dataset has been carefully curated and cleaned to support training at the GPT-2 scale.
The training data originates from the open-source project [nanochat](https://github.com/karpathy/nanochat). We gratefully acknowledge the contributions of Andrej Karpathy and the nanoGPT community.



## Guides

### Energy-based Language Models

A step-by-step turtorial is given in the ['nova/ebt'](https://github.com/openway-ai/nova/tree/main/nanochat/extension/ebt) directory.

### Masked Diffusion Language Models

On the way.


## File structure

```
.
├── LICENSE
├── README.md
├── dev
├── nanochat
├── nova
│   ├── ebt
│   └── mdlm
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   ├── speedrun.sh                 # Train the ~$100 nanochat d20
│   ├── dataset_prepare.sh          # Prepare the dataset
│   ├── install.sh                  # Install dependencies
│   └── run_ebt.sh                  # Train the EBT model
├── scripts
├── tasks
├── tests
└── uv.lock
```

## Research

If you are a researcher and wish to help merge more non-autoregressive models to this repository,


## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~3 hours, but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- This project is primiarily forked from [nanogpt](https://github.com/karpathy/nanoGPT)

<!-- ## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{openway-ai-nova,
  author = {Guanchu Wang},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
``` -->

## License

MIT
