# Energy-based Language Models

This is the reproduction of the ICLR ORAL paper [Energy-based Language Models](https://arxiv.org/abs/2203.02155).

This is the first version of the EBT extended to NanoChat dataset.

### Setup environment

```bash
cd ../../
bash runs/install.sh
cd nova/ebt
```

### Dataset preparation

```bash
cd ../../
bash runs/dataset_prepare.sh
cd nova/ebt
```

### Setting wandb logger

Sign up a free account on [wandb.ai](https://wandb.ai/site), generate an API key, and set it in the '../../runs/run_ebt.sh' as follows:

```bash
export WANDB_API_KEY=<Your API Key>
```

### Training

```bash
cd ../../
bash runs/run_ebt.sh
cd nova/ebt
```

You can check the training logs from the wandb dashboard at `https://wandb.ai/<Your account>/nlp_pretrain`.


## File structure
```
.
├── README.md
├── nanolightning
    ├── __init__.py
    ├── iteratabledataset.py
    ├── iteratabletrainer.py
    ├── torchlightning_function.py
    ├── torchlightning_module.py
    └── torchlightning_trainer.py
├── ar_ebt_adaln.py
├── ar_ebt_default.py
├── ar_ebt_time_embed.py
├── ar_transformer.py
├── collector.py
├── dataset.py
├── eval.py
├── generate.py
├── logger.py
├── metrics.py
├── modeling_ebt.py
├── optimization.py
├── replay_buffer.py
├── tokenizer.py
├── train.py
├── trainer.py
└── utils.py          
```





