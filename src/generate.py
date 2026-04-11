from pathlib import Path
import argparse
import textwrap

import torch

from model import GPT, GPTConfig
from dataset import decode


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    return model, config, stoi, itos, checkpoint


def encode_prompt(prompt: str, stoi: dict[str, int], device: str) -> torch.Tensor:
    safe_prompt = "".join([c for c in prompt if c in stoi])
    if len(safe_prompt) == 0:
        safe_prompt = "\n"

    context = torch.tensor(
        [stoi[c] for c in safe_prompt],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    return context


@torch.no_grad()
def generate_samples(
    model: GPT,
    stoi: dict[str, int],
    itos: dict[int, str],
    device: str,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> list[str]:
    outputs = []

    for _ in range(num_samples):
        context = encode_prompt(prompt, stoi, device)

        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        text_out = decode(generated[0].tolist(), itos)
        outputs.append(text_out)

    return outputs


def save_samples(
    outputs: list[str],
    out_dir: Path,
    checkpoint_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "generation_info.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_name}\n")
        f.write(f"prompt: {repr(prompt)}\n")
        f.write(f"max_new_tokens: {max_new_tokens}\n")
        f.write(f"temperature: {temperature}\n")
        f.write(f"top_k: {top_k}\n")
        f.write(f"num_samples: {len(outputs)}\n")

    for i, text_out in enumerate(outputs, start=1):
        sample_path = out_dir / f"sample_{i}.txt"
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(text_out)


def print_samples(outputs: list[str], max_chars: int = 1200):
    for i, text_out in enumerate(outputs, start=1):
        print("\n" + "=" * 90)
        print(f"SAMPLE {i}")
        print("=" * 90)
        print(text_out[:max_chars])


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a saved GPT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Кад то ",
        help="Initial prompt for generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Number of new characters to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling cutoff",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="generated_samples",
        help="Folder where generated samples will be saved",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.temperature <= 0:
        raise ValueError("temperature must be > 0")

    device = get_device()
    print(f"Using device: {device}")

    model, config, stoi, itos, checkpoint = load_checkpoint(checkpoint_path, device)

    print("\nLoaded checkpoint successfully.")
    print(f"Checkpoint file: {checkpoint_path.name}")
    print(f"Saved step: {checkpoint['step']}")
    print(f"Best val loss stored: {checkpoint['best_val_loss']:.4f}")
    print(f"Config: {checkpoint['config']}")

    outputs = generate_samples(
        model=model,
        stoi=stoi,
        itos=itos,
        device=device,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    save_samples(
        outputs=outputs,
        out_dir=out_dir,
        checkpoint_name=checkpoint_path.name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print_samples(outputs)
    print(f"\nSaved {len(outputs)} samples to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()