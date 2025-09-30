import torch
import torch.nn.functional as F

# TODO: this is probably not being used right now
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

def compare_model_vs_real_probs_subgrammar(model, tokenizer, test_sequences_with_probs, device):
    """
    Now expects test_sequences_with_probs to be a list of (sequence_str, real_log_prob) tuples.
    We unpack both, but only recompute model_log_prob and then compare to the stored real_log_prob.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for seq, real_log_prob in test_sequences_with_probs:
            start, end, text = seq
            encoded = tokenizer.encode(text, return_tensors="pt").to(device)
            model_log_prob = 0.0

            length = end - start + 1
            start += 1
            for i in range(length):
                input_ids = encoded[:, : start + i]
                target_id = encoded[:, start + i].item()
                logits, _ = model(input_ids)
                log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
                token_log_prob = log_probs[0, target_id].item()
                model_log_prob += token_log_prob

            diff = abs(model_log_prob - real_log_prob)

            results.append({
                "text": text,
                "log_prob_model": model_log_prob,
                "log_prob_real": real_log_prob,
                "abs_logprob_diff": diff,
            })

    return results

# these are currently not being used

# def calculate_accuracy(generated_sequences, train_sequences, grammar_name, tokenizer):
#     """Calculate the accuracy of generated sequences against valid sequences."""
#     train_set = set(tuple(seq) for seq in train_sequences)
#     accuracy = sum(validate(tokenizer, seq, grammar_name) for seq in generated_sequences) / len(generated_sequences)
#     train_overlap = sum(1 for seq in generated_sequences if tuple(seq) in train_set) / len(generated_sequences)
#     return accuracy, train_overlap

# def evaluate_generated_sequences(model, tokenizer, training_sequences, grammar_name, test_sequences, device, num_samples=50, max_length=128):
#     """Evaluate generated sequences on accuracy and perplexity."""
#     generated_sequences = generate_and_score_sequences(model, tokenizer, num_samples, max_length, device)
#     accuracy, train_overlap = calculate_accuracy(generated_sequences, training_sequences, grammar_name, tokenizer)
#     res = compare_model_vs_real_probs(model, tokenizer, test_sequences, device)
#     return generated_sequences, accuracy, train_overlap, res