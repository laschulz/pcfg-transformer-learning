import random
from transformers import GPT2Tokenizer
import os
import pickle

MAX_SEQUENCE_LENGTH = 10
DATASET_SIZE = 10000

# probably have differenet classes for different types of grammars
class PCFG:   
    def __init__(self, rules):
        """
        rules: dict of the form
        {
            'S': [(['NP', 'VP'], 0.9), (['VP'], 0.1)],
            'NP': [(['Det', 'N'], 1.0)],
            ...
        }
        """

        self.rules = rules
        self.name = 'basic'


class LanguageGenerator:
    def __init__(self, pfcg):
        self.rules = pfcg.rules


    def generate_sequence(self, max_length):
        sequence = []
        stack = ['S']  # Start with the start symbol

        while stack and len(sequence) < max_length:
            symbol = stack.pop()
            if symbol in self.rules:
                productions, probs = zip(*self.rules[symbol])
                production = random.choices(productions, weights=probs, k=1)[0]
                stack.extend(reversed(production))
            else:
                sequence.append(symbol)
        return sequence

    def generate_sequences(self, num_sequences, max_length):
        return [self.generate_sequence(max_length) for _ in range(num_sequences)]



def save_dataset(sequences, dataname):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    all_tokens = []
    for seq in sequences:
        text = ' '.join(seq)
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    # Save as binary token array
    os.makedirs(dataname, exist_ok=True)
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(dataname, f'train.bin'))

    # Save metadata
    meta = {
        'vocab_size': tokenizer.vocab_size,
    }
    with open(os.path.join(dataname, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved {len(all_tokens)} tokens to {dataname}/train.bin")



rules = {
        'S': [(['NP', 'VP'], 0.7), (['VP'], 0.3)],
        'NP': [(['Det', 'N'], 1.0)],
        'VP': [(['V', 'NP'], 0.8), (['V'], 0.2)],
        'Det': [(['a'], 0.5), (['the'], 0.5)],
        'N': [(['cat'], 0.5), (['dog'], 0.5)],
        'V': [(['chases'], 1.0)]
    }

pfcg = PCFG(rules)
generator = LanguageGenerator(pfcg)
sequences = generator.generate_sequences(DATASET_SIZE, MAX_SEQUENCE_LENGTH)
save_dataset(sequences, 'data/pcfg')