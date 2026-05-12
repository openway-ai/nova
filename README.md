# OpenEBM

**Open-source Energy-Based language Models for non-autoregressive generation and reasoning.**

OpenEBM is a research codebase for training and evaluating Energy-Based
Transformers (EBT) and other non-autoregressive language models at scale. It
extends the minimalist [nanochat](https://github.com/karpathy/nanochat)
training stack with MCMC-style latent refinement, SFT and reasoning
benchmarks (GSM8K, Sudoku, SpellingBee), and a PyTorch Lightning trainer
specialised for energy-based objectives.

> The `openebm/elm/` sub-package contains the Energy-Based Language Model
> (ELM) training stack. The rest of the repository (`nanochat/`) is vendored
> from the upstream project as a read-only dependency.

---

## Features

- **Energy-Based Transformer** — three variants are provided out of the box:
  `time_embed`, `adaln` and `default`. All variants support learnable MCMC
  step sizes, random-noise / denoising initial conditions, and replay
  buffers.
- **Multi-objective SFT** — a `TaskMixture` pipeline combining SmolTalk,
  MMLU, GSM8K, SpellingBee, CustomJSON and Sudoku (v1 SATNet and v2 RRN).
- **Sudoku reasoning benchmark** — train / validate with RRN, evaluate on
  SATNet (or vice versa), with digit-relabel, band / stack permutation and
  transposition augmentations.
- **Distributed training** — `torchrun` launch scripts, Muon + AdamW hybrid
  optimizer, dynamic weight decay, cosine warmdown, disk-aware checkpointing.
- **Chat & evaluation scripts** — `chat_ebt.py`, `chat_ebt_web.py` and
  `ebt_core_eval.py` for interactive inference and core-capability eval.

---

## Installation

OpenEBM targets **Python >= 3.10** and CUDA 12.8. We recommend
[uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/openway-ai/OpenEBM.git
cd OpenEBM
uv sync --extra gpu     # or `--extra cpu` for a CPU-only install
```

Set your Weights & Biases key:

```bash
export WANDB_API_KEY=<your key>
```

---

## Quickstart

### Pre-train EBT on the NanoChat dataset

```bash
bash openebm/elm/runs/run_ebt.sh
```

### Supervised fine-tuning (SFT)

```bash
bash openebm/elm/runs/run_ebt_sft.sh
```

### Sudoku reasoning SFT

```bash
bash openebm/elm/runs/run_ebt_sudoku_sft.sh
```

### Interactive chat

```bash
python -m openebm.elm.scripts.chat_ebt --ckpt <path/to/ckpt>
```

---

## Directory layout

```
.
├── openebm/
│   └── elm/                         # Energy-Based Language Model stack
│       ├── ar_ebt_*.py              # EBT variants (time_embed / adaln / default)
│       ├── ar_transformer.py        # Base LLaMA-style transformer
│       ├── modeling_ebt.py          # EBT model wrapper
│       ├── trainer.py               # PyTorch Lightning training module
│       ├── train.py                 # CLI entry point
│       ├── generate.py              # Generation / MCMC refinement utilities
│       ├── dataset*.py              # Pretraining and SFT datasets
│       ├── data/                    # Sudoku datasets and evaluators
│       ├── nanolightning/           # Minimal Lightning fork
│       └── scripts/                 # chat / eval / web demo scripts
├── nanochat/                        # Vendored upstream dependency (read-only)
├── docs/                            # Sphinx documentation (openebm.elm only)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Documentation

Sphinx documentation for the `openebm.elm` package lives in `docs/`:

```bash
cd docs && make html
open _build/html/index.html
```

The docs only index the `openebm.elm` sub-package; `nanochat/` is excluded
by design.

---

## Contributing

We welcome contributions that extend the non-autoregressive language-model
stack. Please open an issue describing your proposed change before sending
a pull request. Guidelines:

- Run `ruff check openebm/elm` and `mypy openebm/elm` before opening a PR.
- Follow the Sphinx reST docstring style used across the codebase.
- Do not modify files under `nanochat/` — that directory is vendored.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Citation

If you use OpenEBM in academic work, please cite both this repository and
the underlying EBT paper:

```bibtex
@inproceedings{ebt2025,
  title     = {Energy-Based Transformers are Scalable Learners and Thinkers},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {Oral}
}
```

---

## Acknowledgements

OpenEBM is built on top of
[nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy. We
gratefully acknowledge the nanoGPT / nanochat community for releasing such
an accessible foundation.

## License

[MIT](LICENSE).
