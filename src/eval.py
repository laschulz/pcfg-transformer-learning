import torch
import torch.nn.functional as F
from generate_pcfg import validate

def generate_and_score_sequences(model, tokenizer, num_samples, max_length, device):
    """
    Generate sequences from the model and compute their log-probabilities.
    """
    model.eval()
    results = []

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(num_samples):
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
            output, _ = model.generate(input_ids, max_new_tokens=max_length, eos_token_id=eos_token_id, eos_prob_threshold=0.9)

            # Remove batch dim and compute log-prob
            sequence = output[0]

            results.append(
                tokenizer.decode(sequence, skip_special_tokens=True),
            )

    return results


def score_known_sequences(model, tokenizer, sequences, device):
    model.eval()
    results = []
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    with torch.no_grad():
        for seq in sequences:
            encoded = tokenizer.encode(bos_token + " " + seq + " " + eos_token, return_tensors="pt").to(device)

            sequence_log_prob = 0.0
            for i in range(encoded.size(1) - 1):
                # Input: tokens up to position i
                input_ids = encoded[:, :i+1]
                # Target: token at position i+1
                target_id = encoded[:, i+1].item()
                
                # Get logits (will be for the last position only)
                logits, _ = model(input_ids)
                # Apply softmax to get probabilities
                log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
                # Get log prob for the target token
                token_log_prob = log_probs[0, target_id].item()
                sequence_log_prob += token_log_prob
            
            results.append({
                "text": seq,
                "log_prob": sequence_log_prob,
            })

    return results

def calculate_accuracy(generated_sequences, train_sequences, grammar_name):
    """Calculate the accuracy of generated sequences against valid sequences."""
    train_set = set(tuple(seq) for seq in train_sequences)
    accuracy = sum(validate(seq, grammar_name) for seq in generated_sequences) / len(generated_sequences)
    train_overlap = sum(1 for seq in generated_sequences if tuple(seq) in train_set) / len(generated_sequences)
    return accuracy, train_overlap

def evaluate_generated_sequences(model, tokenizer, training_sequences, grammar_name, test_sequences, device, num_samples=50, max_length=256):
    """Evaluate generated sequences on accuracy and perplexity."""
    generated_sequences = generate_and_score_sequences(model, tokenizer, num_samples, max_length, device)
    #print(generated_sequences)
    accuracy, train_overlap = calculate_accuracy(generated_sequences, training_sequences, grammar_name)
    #print(f"Accuracy: {accuracy:.4f}, Train Overlap: {train_overlap:.4f}")
    res = score_known_sequences(model, tokenizer, test_sequences, device)
    return generated_sequences, accuracy, train_overlap, res