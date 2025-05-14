import torch

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)