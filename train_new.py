import torch
import numpy as np
import os
import pickle
from transformers import GPT2Tokenizer

from model import GPT, GPTConfig
    
# ----- Config -----
data_dir = 'data'
dataset = 'basic'
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
checkpoint_every = 1
num_epochs = 10
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# ----- Load Dataset -----
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, dataset, f'{split}.bin'), dtype=np.uint32, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ----- Load vocab size from meta.pkl -----
with open(os.path.join(data_dir, dataset, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
bos_token_id = meta['bos_token_id']

# ----- Initialize the model -----
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    vocab_size=vocab_size,
    dropout=dropout,
    bias=bias
)
model = GPT(GPTConfig(**model_args)).to(device)

# ----- Optimizer -----
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# ----- Training Loop -----
tokens_per_epoch = len(np.memmap(os.path.join(data_dir, dataset, 'train.bin'), dtype=np.uint16, mode='r'))
iters_per_epoch = tokens_per_epoch // (batch_size * block_size)
for epoch in range(num_epochs):
    for _ in range(iters_per_epoch):
        # Train step
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    #save checkpoint and evaluate
    if (epoch+1) % checkpoint_every == 0:
        checkpoint_path = os.path.join(data_dir, dataset, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # have model generate some text and save it in a txt file
        model.eval()
        start_token = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.generate(start_token, max_new_tokens=30)
            decoded = tokenizer.decode(out[0].tolist())
            print(f"[Epoch {epoch}] Sample Output: {decoded}")
        model.train()