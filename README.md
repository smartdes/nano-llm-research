# Nano-LLM Research

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Note**: This is my first project exploring deep learning and Large Language Models (LLMs) from scratch!

A clean, modular, and educational implementation of an autoregressive **Decoder-Only Transformer (GPT)** built from scratch in PyTorch. 

Designed for experimentation, research, and learning core Large Language Model (LLM) mechanisms: Multi-Head Causal Self-Attention, Residual Connections, Pre-Layer Normalization, character-level tokenization, and temperature-controlled top-$k$ autoregressive sampling.

---

## Architecture Overview

```
                      Input Token Indices (B, T)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       Token Embedding (V → C)        Position Embedding (T → C)
                  └───────────────┬───────────────┘
                                  ▼
                              Sum (B, T, C)
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Transformer Block (Repeated N times)             │
        │                                                   │
        │   ┌───► LayerNorm ──► Causal Self-Attention ──+   │
        │   │                                           │   │
        │   x ◄─────────────────────────────────────────┘   │
        │   │                                               │
        │   └───► LayerNorm ──► Feed-Forward (GELU) ───+    │
        │                                              │    │
        │   x ◄────────────────────────────────────────┘    │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
                          Final LayerNorm
                                  ▼
                        Linear Head (C → V)
                                  ▼
                       Logits / Cross-Entropy
```

### Key Mathematical Components

- **Scaled Dot-Product Attention with Causal Masking**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
  where $M_{i,j} = -\infty$ for $j > i$ (preventing attending to future tokens).
- **Multi-Head Projection**: Linear projections across $h$ heads of dimension $d_k = d_{\text{model}} / h$.
- **Feed-Forward Network (FFN)**: Two-layer MLP with $\text{GELU}$ activation expanding to $4 \times d_{\text{model}}$.

---

## Project Structure

```text
nano-llm-research/
├── configs/
│   └── config.yaml          # Model & training hyperparameters
├── data/
│   ├── dataset.py           # TextDataset class with character tokenizer
│   └── input.txt            # Preprocessed training corpus
├── models/
│   └── mini_gpt.py          # CausalSelfAttention, Block, and MiniGPT
├── checkpoints/             # Model weight checkpoints
│   └── .gitkeep
├── main.py                  # Main CLI entry point (train & generate)
├── requirements.txt         # Dependency specifications
├── .gitignore               # Standard Python & PyTorch exclusions
└── README.md                # Documentation & quickstart
```

---

## Quickstart

### 1. Prerequisites & Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/smartdes/nano-llm-research.git
cd nano-llm-research

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training & Text Generation

Run the default training pipeline on the provided corpus:

```bash
python main.py
```

### 3. CLI Options & Text Generation

The entry point supports full customization via CLI arguments:

```bash
# Prompt conditioning with top-k sampling and temperature adjustment
python main.py --prompt "Through her courage" --temperature 0.75 --top-k 40 --max-tokens 400

# Specify execution device (auto, cuda, mps, or cpu)
python main.py --device cuda --max-iters 2000

# Use a custom configuration file
python main.py --config configs/custom_config.yaml
```

#### Available CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/config.yaml` | Path to YAML config file |
| `--prompt` | `str` | `""` | Seed string to prime text generation |
| `--max-tokens` | `int` | `300` | Number of tokens to autoregressively sample |
| `--temperature`| `float` | `0.8` | Softmax temperature ($<1.0$ sharper, $>1.0$ more uniform) |
| `--top-k` | `int` | `40` | Top-$k$ filtering ($0$ or negative to disable) |
| `--device` | `str` | `auto` | Execution device: `auto`, `cuda`, `mps`, or `cpu` |
| `--max-iters` | `int` | `None` | Override training iterations from config |
| `--seed` | `int` | `1337` | Random seed for reproducibility |

---

## Configuration Reference (`configs/config.yaml`)

```yaml
model:
  vocab_size: 65         # Updated dynamically to match dataset vocabulary
  d_model: 128           # Embedding & hidden representation dimension
  n_heads: 4             # Number of self-attention heads
  n_layers: 6            # Number of Transformer blocks
  block_size: 64         # Context window length (maximum sequence length)

training:
  batch_size: 32         # Number of parallel sequences per gradient step
  learning_rate: 0.0008  # AdamW initial learning rate
  max_iters: 1500        # Total optimization steps
  eval_interval: 300     # Evaluation and logging frequency
  eval_iters: 20         # Number of batches sampled for validation loss estimation
  data_path: "data/input.txt"
  save_checkpoint: true
  checkpoint_path: "checkpoints/mini_gpt.pt"
```

---

## Dataset

The default sample dataset located at `data/input.txt` is a curated historical address (Barack Obama's 2005 Senate Floor tribute to Rosa Parks), formatted into clean paragraphs without OCR or citation artifacts.

### Using Custom Data
To train on your own corpus:
1. Replace `data/input.txt` (or point `data_path` in `configs/config.yaml` to your `.txt` file).
2. The tokenizer will automatically detect the unique character set and adjust `vocab_size`.

---

## References & Acknowledgments

- Vaswani et al., *"Attention Is All You Need"* (2017) [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)]
- Radford et al., *"Language Models are Unsupervised Multitask Learners"* (GPT-2)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) for architectural inspiration.

---

## License

This project is licensed under the [MIT License](LICENSE).
