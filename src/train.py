import torch

from dataset import load_text, build_vocab, encode, split_data, get_batch
from model import GPT, GPTConfig


# --------------------------------------------------
# Toggle this:
# True  -> tiny overfit debugging test
# False -> normal training settings
# --------------------------------------------------
OVERFIT_TEST = True


def main():
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    block_size = 128
    batch_size = 4 if OVERFIT_TEST else 32
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.1
    learning_rate = 3e-4
    max_steps = 200 if OVERFIT_TEST else 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Overfit test: {OVERFIT_TEST}")

    # -----------------------------
    # Load and prepare data
    # -----------------------------
    text = load_text()
    stoi, itos, vocab_size = build_vocab(text)

    data = encode(text, stoi)
    train_data, val_data = split_data(data)

    # Tiny subset for debugging / overfitting
    if OVERFIT_TEST:
        train_data = train_data[:500]

    print(f"Vocab size: {vocab_size}")
    print(f"Train data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")

    # -----------------------------
    # Build model
    # -----------------------------
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        attn_impl="sdpa",  # later you can compare with "manual"
    )

    model = GPT(config).to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # -----------------------------
    # Training loop
    # -----------------------------
    model.train()

    for step in range(max_steps):
        x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size)
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == max_steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # -----------------------------
    # Final check on one batch
    # -----------------------------
    x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size)
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)
    print(f"\nFinal loss on sampled batch: {loss.item():.4f}")


if __name__ == "__main__":
    main()