import sys
print(sys.path)
from configuration.config import ROOT_DIR


def train():
    print(f"root_dir:{ROOT_DIR}")
if __name__ == "__main__":
    train()