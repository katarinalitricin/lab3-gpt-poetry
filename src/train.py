from pathlib import Path
from dataclasses import asdict

import torch

from dataset import load_text, build_vocab, encode, split_data, get_batch, decode
from model import GPT, GPTConfig


# ============================================================
# Config

block_size = 256
batch_size = 32
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.2

learning_rate = 3e-4
max_iters = 10000
eval_interval = 500
eval_iters = 200
sample_interval = 500
save_interval = 2000
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

out_dir = Path("outputs/checkpoints")
sample_dir = Path("outputs/samples")
log_path = Path("outputs/losses.pt")

out_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)


def estimate_loss(model, train_data, val_data):
    """Estimate average train/val loss over a few batches."""
    model.eval()
    out = {}

    for split_name, split_data_ in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data_, block_size=block_size, batch_size=batch_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()

    model.train()
    return out


@torch.no_grad()
def generate_sample(model, stoi, itos, step, prompt="Кад то ", max_new_tokens=400):
    """Generate and save one sample during training."""
    model.eval()

    safe_prompt = "".join([c for c in prompt if c in stoi])
    if len(safe_prompt) == 0:
        safe_prompt = "\n"

    context = torch.tensor(
        [stoi[c] for c in safe_prompt],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
    )

    text_out = decode(generated[0].tolist(), itos)

    sample_path = sample_dir / f"sample_step_{step}.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(text_out)

    print("\n--- SAMPLE ---")
    print(text_out[:1200])
    print("--------------\n")

    model.train()


def save_checkpoint(model, optimizer, config, stoi, itos, step, best_val_loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "stoi": stoi,
        "itos": itos,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    checkpoint_path = out_dir / f"model_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Data

    text = load_text()
    stoi, itos, vocab_size = build_vocab(text)

    data = encode(text, stoi)
    train_data, val_data = split_data(data)

    print(f"Vocab size: {vocab_size}")
    print(f"Train data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")

    # --------------------------------------------------------
    # Model

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        attn_impl="sdpa",   # later compare with "manual"
    )

    model = GPT(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    # added: loss logs
    steps_log = []
    train_log = []
    val_log = []

    # --------------------------------------------------------
    # Training loop

    model.train()

    for step in range(max_iters):
        x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:5d} | train loss {loss.item():.4f}")

        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_loss = losses["train"]
            val_loss = losses["val"]

            print(
                f"[eval] step {step:5d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )

            # added: save logs
            steps_log.append(step)
            train_log.append(train_loss)
            val_log.append(val_loss)
            torch.save(
                {
                    "steps": steps_log,
                    "train_loss": train_log,
                    "val_loss": val_log,
                },
                log_path,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, config, stoi, itos, step, best_val_loss)
                print(f"  -> saved new best checkpoint (val loss {best_val_loss:.4f})")

        if step % sample_interval == 0 and step > 0:
            generate_sample(model, stoi, itos, step, prompt="Кад то ")

        if step % save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, config, stoi, itos, step, best_val_loss)
            print("  -> saved periodic checkpoint")

    # added: final checkpoint
    save_checkpoint(model, optimizer, config, stoi, itos, max_iters, best_val_loss)
    print("Saved final checkpoint.")

    print("Training complete.")


if __name__ == "__main__":
    main()