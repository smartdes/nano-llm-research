import os
import argparse
import yaml
import torch
from data.dataset import TextDataset
from models.mini_gpt import MiniGPT

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nano-LLM: Train and generate text with a lightweight decoder-only Transformer."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional text prompt to condition generation."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Number of tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower = more deterministic, higher = more creative)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k filtering for text generation (0 or negative to disable)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device."
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override training iterations specified in config."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def get_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested_device)

@torch.no_grad()
def estimate_loss(model, dataset, batch_size: int, eval_iters: int, device: torch.device):
    """Estimate loss on train and validation splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = dataset.get_batch(split, batch_size, device=device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Load Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device(args.device)
    print(f"[*] Initialized Nano-LLM on device: {device}")

    # 2. Load Dataset
    data_path = config['training'].get('data_path', 'data/input.txt')
    dataset = TextDataset(data_path, config['model']['block_size'])
    config['model']['vocab_size'] = dataset.vocab_size

    print(f"[*] Loaded dataset from '{data_path}'")
    print(f"    - Vocabulary size : {dataset.vocab_size} characters")
    print(f"    - Total corpus    : {len(dataset.text):,} characters")
    print(f"    - Train tokens    : {len(dataset.train_data):,}")
    print(f"    - Val tokens      : {len(dataset.val_data):,}")

    # 3. Initialize Model and Optimizer
    model = MiniGPT(config['model']).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Model parameters : {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )

    max_iters = args.max_iters or config['training'].get('max_iters', 1500)
    eval_interval = config['training'].get('eval_interval', 300)
    eval_iters = config['training'].get('eval_iters', 20)
    batch_size = config['training'].get('batch_size', 32)

    # 4. Training Loop
    print(f"[*] Starting training for {max_iters} iterations...")
    model.train()
    for iter_step in range(max_iters + 1):
        if iter_step % eval_interval == 0 or iter_step == max_iters:
            losses = estimate_loss(model, dataset, batch_size, eval_iters, device)
            print(
                f"    Step {iter_step:5d} / {max_iters} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f}"
            )

        if iter_step == max_iters:
            break

        xb, yb = dataset.get_batch('train', batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    # 5. Save Checkpoint (optional)
    if config['training'].get('save_checkpoint', False):
        checkpoint_path = config['training'].get('checkpoint_path', 'checkpoints/mini_gpt.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'vocab_size': dataset.vocab_size,
            },
            checkpoint_path
        )
        print(f"[*] Checkpoint saved to '{checkpoint_path}'")

    # 6. Text Generation
    print("\n" + "=" * 50)
    print("                 GENERATED OUTPUT")
    print("=" * 50)
    model.eval()

    if args.prompt:
        encoded_prompt = dataset.encode(args.prompt)
        if not encoded_prompt:
            encoded_prompt = [0]
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    top_k = args.top_k if args.top_k > 0 else None
    generated_indices = model.generate(
        context,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k
    )

    generated_text = dataset.decode(generated_indices[0].tolist())
    print(generated_text)
    print("=" * 50)

if __name__ == "__main__":
    main()
