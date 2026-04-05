from pathlib import Path

DATA_PATH = Path("data/raw/poems.txt")

def load_text():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__mainpython -m venv .venv__":
    text = load_text()
    print("Loaded characters:", len(text))
    print(text[:300])