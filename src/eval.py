import torch
import torch.nn.functional as F
from generate_pcfg import validate

#this is a bit weird
def generate_and_score_sequences(model, tokenizer, num_samples=50, max_length=256):
    """
    Generate sequences from the model and compute their log-probabilities.
    """
    model.eval()
    results = []

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    device = next(model.parameters()).device #adjust this

    with torch.no_grad():
        for _ in range(num_samples):
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
            output, logits = model.generate(input_ids, max_new_tokens=max_length, eos_token_id=eos_token_id, eos_prob_threshold=0.9)

            # Remove batch dim and compute log-prob
            sequence = output[0]
            target_seq = sequence[1:].unsqueeze(0)  # exclude first token for target

            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(2, target_seq.unsqueeze(-1)).squeeze(-1)
            sequence_log_prob = selected_log_probs.sum().item()

            results.append({
                "text": tokenizer.decode(sequence, skip_special_tokens=True),
                "log_prob": sequence_log_prob,
            })

    return results

def calculate_accuracy(generated_sequences, train_sequences, grammar_name):
    """Calculate the accuracy of generated sequences against valid sequences."""
    train_set = set(tuple(seq) for seq in train_sequences)
    accuracy = sum(validate(seq, grammar_name) for seq in generated_sequences) / len(generated_sequences)
    train_overlap = sum(1 for seq in generated_sequences if tuple(seq) in train_set) / len(generated_sequences)
    return accuracy, train_overlap

def evaluate_generated_sequences(model, tokenizer, training_sequences, grammar_name, num_samples=10, max_length=256):
    """Evaluate generated sequences on accuracy and perplexity."""
    generated_sequences = generate_and_score_sequences(model, tokenizer, num_samples, max_length)
    print(generated_sequences)
    accuracy, train_overlap = calculate_accuracy(generated_sequences, training_sequences, grammar_name)
    print(f"Accuracy: {accuracy:.4f}, Train Overlap: {train_overlap:.4f}")
    return accuracy, train_overlap