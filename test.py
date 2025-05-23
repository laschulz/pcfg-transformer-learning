import torch
import torch.nn.functional as F

#this is a bit weird
def generate_and_score_sequences(model, tokenizer, num_samples=10, max_length=256):
    """
    Generate sequences from the model and compute their log-probabilities.
    """
    model.eval()
    results = []

    bos_token_id = tokenizer.bos_token_id
    device = next(model.parameters()).device #adjust this

    with torch.no_grad():
        for _ in range(num_samples):
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.uint16).to(device)
            output = model.generate(input_ids, max_length=max_length, do_sample=True)

            # Remove batch dim and compute log-prob
            sequence = output[0]
            input_seq = sequence[:-1].unsqueeze(0)  # exclude last token for input
            target_seq = sequence[1:].unsqueeze(0)  # exclude first token for target

            logits = model(input_seq).logits
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(2, target_seq.unsqueeze(-1)).squeeze(-1)
            sequence_log_prob = selected_log_probs.sum().item()

            results.append({
                "text": tokenizer.decode(sequence, skip_special_tokens=True),
                "log_prob": sequence_log_prob,
            })

    return results

def calculate_accuracy(generated_sequences, valid_sequences, train_sequences):
    """Calculate the accuracy of generated sequences against valid sequences."""
    valid_set = set(tuple(seq) for seq in valid_sequences)
    train_set = set(tuple(seq) for seq in train_sequences)

    accuracy = sum(1 for seq in generated_sequences if tuple(seq) in valid_set) / len(generated_sequences)
    train_overlap = sum(1 for seq in generated_sequences if tuple(seq) in train_set) / len(generated_sequences)

    return {
        "accuracy": accuracy,
        "train_overlap": train_overlap
    }

def evaluate_generated_sequences(model, tokenizer, valid_sequences, num_samples=10, max_length=256):
    """Evaluate generated sequences on accuracy and perplexity."""
    generated_sequences = generate_and_score_sequences(model, tokenizer, num_samples, max_length)
    accuracy = calculate_accuracy(generated_sequences, valid_sequences)
    return accuracy