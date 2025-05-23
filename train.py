import torch
import numpy as np
import os
import pickle
from transformers import GPT2Tokenizer

from model import GPT, TwoLayerTransformerConfig
    
# ----- Config -----
data_dir = 'data'
dataset = 'LinearRecursion'
batch_size = 8
learning_rate = 6e-4
checkpoint_every = 1
num_epochs = 10
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
early_stopping = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
model = TwoLayerTransformerConfig.to(device)

# ----- Load Dataset -----
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, dataset, f'{split}.bin'), dtype=np.uint32, mode='r')
    ix = torch.randint(len(data) - model.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+model.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+model.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ----- Load vocab size from meta.pkl -----
with open(os.path.join(data_dir, dataset, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
bos_token_id = meta['bos_token_id']

# ----- Optimizer -----
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# ----- Training Loop -----
tokens_per_epoch = len(np.memmap(os.path.join(data_dir, dataset, 'train.bin'), dtype=np.uint16, mode='r'))
iters_per_epoch = tokens_per_epoch // (batch_size * model.block_size)
best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0
for epoch in range(num_epochs):
    model.train()
    for _ in range(iters_per_epoch):
        # Train step
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(10):  # evaluate on 10 batches
            X_val, Y_val = get_batch('val')
            _, val_loss = model(X_val, Y_val)
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    #save checkpoint and evaluate
    if (epoch+1) % checkpoint_every == 0:
        checkpoint_path = os.path.join(data_dir, dataset, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # have model generate some text and save it in a txt file
        start_token = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.generate(start_token, max_new_tokens=30)
            decoded = tokenizer.decode(out[0].tolist())
            print(f"[Epoch {epoch}] Sample Output: {decoded}")