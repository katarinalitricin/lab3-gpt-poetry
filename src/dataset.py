import os
import torch

def load_text(path="data/raw/poems.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

#Build vocab
def build_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos, vocab_size

#Encode Decode
def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(indices, itos):
    return "".join([itos[i] for i in indices])

#Train/validation split
def split_data(data, split_ratio=0.9):
    n = int(len(data) * split_ratio)
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])

    return x, y

if __name__ == "__main__":
    text = load_text()
    stoi, itos, vocab_size = build_vocab(text)

    data = encode(text, stoi)
    train_data, val_data = split_data(data)

    x, y = get_batch(train_data, block_size=128, batch_size=4)

    print("Vocab size:", vocab_size)
    print("Batch shape:", x.shape)