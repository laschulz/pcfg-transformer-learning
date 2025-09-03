# 1. train model
# 2. load model
# 3. run 
from model import GPT
import torch
import torch.nn.functional as F
from train import map_model_name
import argparse
from transformers import PreTrainedTokenizerFast


SEQUENCES=["sL1 sL2 cond eL2", #same
           "sL1 sL2 not cond eL2", #same
           "sL1 sL2 cond eL2 sL2 not cond eL2", #1 loop
           "sL1 sL2 cond and cond eL2 sL2 cond eL2", # 1 loop
           "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2", # 2 loops
           "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2", # 2 loops
           "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2", # 3 loops
           "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2", # 3 loops
           "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2", # 4 loops
            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2", # 4 loops
            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond eL2", # 5 loops
            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2", # 5 loops
            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2", # 6 loops
            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2" # 6 loops
            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2 sL2 cond eL2", # 7 loops
            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2" # 7 loops
            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2 sL2 cond eL2 sL2 cond and cond eL2", # 8 loops
            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2" # 8 loops
            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2 sL2 cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2", # 9 loops
            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 not cond and cond and condeL2" # 9 loops
]

OTHER_SEQUENCES=["sL1_2 sL2 cond eL2", #completely different
                "sL1 sL2 cond eL2 action sL2_3 a a a", # something else
]

def collect_activations(model, tokenizer, sequences, agg=None):
    # 1) Prepare empty buffers
    n_layers = len(model.transformer.h)
    acts = {f"attn_{i}": [] for i in range(n_layers)}
    acts.update({f"mlp_{i}": [] for i in range(n_layers)})

    # 2) Define hook factory
    def get_hook(key):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            B, L, D = out.shape
            if agg is None:
                rep = out.reshape(B*L, D)
            elif agg == "mean":
                rep = out.mean(dim=1)  # (B, D) -> (1, D)
            elif agg == "mean_unit":
                z = out / (out.norm(dim=-1, keepdim=True) + 1e-8)  # per-token unit
                rep = z.mean(dim=1)  
            acts[key].append(rep.detach().cpu())
        return hook

    # 3) Register hooks
    hooks = []
    for i, block in enumerate(model.transformer.h):
        # be robust to naming
        attn_mod = getattr(block, "attn", None) or getattr(block, "mha", None) or getattr(block, "self_attn", None)
        mlp_mod  = getattr(block, "mlp",  None) or getattr(block, "ffn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(get_hook(f"attn_{i}")))
        if mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(get_hook(f"mlp_{i}")))

    # 4) Run forward passes
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt")
            input_ids = enc["input_ids"]                    # shape (1, L)
            _ = model(input_ids)  

    # 5) Remove hooks
    for h in hooks:
        h.remove()

    # 6) Concatenate into single tensor per layer
    for k in list(acts.keys()):
        acts[k] = torch.cat(acts[k], dim=0)

    return acts

def collect_activations_per_input(model, tokenizer, inputs, agg="tokens"):
    """
    Capture layer-wise activations for ONE model over a list of input strings.
    Returns: dict[str -> dict[layer_key -> tensor]]
      - if agg == "tokens": tensor is [T_i, D] for each input i (no padding)
      - if agg == "last":   tensor is [D]     (last non-pad token per input)
      - if agg == "mean":   tensor is [D]     (mean over tokens)
      - if agg == "mean_unit": [D]            (mean of unit-normalized tokens)
    """
    device = next(model.parameters()).device
    model.eval()

    # Tokenize together (padding so we can do one forward pass)
    batch = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    B, T = batch["input_ids"].shape
    lengths = batch["attention_mask"].sum(dim=1)  # [B] number of real tokens

    # Prepare storage for raw hook outputs (batched), then we'll split per-input
    n_layers = len(model.transformer.h)
    raw = {f"attn_{i}": None for i in range(n_layers)}
    raw.update({f"mlp_{i}": None for i in range(n_layers)})

    def get_hook(key):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if x.ndim == 2:  # [T,D] -> [1,T,D]
                x = x.unsqueeze(0)
            raw[key] = x.detach().cpu()  # [B, T, D]
        return hook

    # Register hooks (robust to naming differences)
    hooks = []
    for i, block in enumerate(model.transformer.h):
        attn_mod = getattr(block, "attn", None) or getattr(block, "mha", None) or getattr(block, "self_attn", None)
        mlp_mod  = getattr(block, "mlp",  None) or getattr(block, "ffn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(get_hook(f"attn_{i}")))
        if mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(get_hook(f"mlp_{i}")))

    with torch.no_grad():
        _ = model(**batch, use_cache=False)

    for h in hooks:
        h.remove()

    # Now split the batched tensors into per-input tensors and apply agg
    per_input = {s: {} for s in inputs}
    for key, X in raw.items():
        if X is None:
            continue  # module might not exist
        # X: [B, T, D]
        for i, s in enumerate(inputs):
            Li = int(lengths[i].item())
            Xi = X[i, :Li, :]  # strip padding for this input -> [Li, D]
            if agg == "tokens":
                rep = Xi
            elif agg == "last":
                rep = Xi[-1]
            elif agg == "mean":
                rep = Xi.mean(dim=0)
            elif agg == "mean_unit":
                Zi = Xi / (Xi.norm(dim=-1, keepdim=True) + 1e-8)
                rep = Zi.mean(dim=0)
            else:
                raise ValueError(f"Unknown agg={agg}")
            per_input[s][key] = rep  # [Li, D] or [D]

    return per_input

def get_logits(model, tokenizer, sequences, device):
    model.eval()
    results = []

    with torch.no_grad():
        for seq in sequences:
            # Encode string with BOS/EOS
            encoded = tokenizer.encode(seq,
                return_tensors="pt"
            ).to(device)

            # Forward pass
            logits, _ = model(encoded)
            log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
            results.append(log_probs.detach().cpu())
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    config = map_model_name(args.model)
    model = GPT(config)

    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer_path = f'{args.tokenizer_path}/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token="<|bos|>", eos_token="<|eos|>")

    logits = get_logits(model, tokenizer, SEQUENCES, device)
    print(logits)


    # activations = collect_activations_per_input(model, tokenizer, SEQUENCES, agg='mean')
    # for layer, acts in activations.items():
    #     print(f"Layer: {layer}, Activations shape: {acts.shape}")

if __name__ == "__main__":
    main()