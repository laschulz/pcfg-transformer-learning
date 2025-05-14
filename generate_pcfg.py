import random
from transformers import GPT2Tokenizer
import os
import pickle
import numpy as np

MAX_SEQUENCE_LENGTH = 10
DATASET_SIZE = 10000

# probably have differenet classes for different types of grammars
class PCFG_basic:   
    def __init__(self):
        
        self.rules = {
            "S": [(["NP", "VP"], 1.0)],
            "NP": [(["Det", "N"], 0.4), (["Det", "Adj", "N"], 0.3), (["NP", "PP"], 0.3)],
            "VP": [(["Vtrans", "NP"], 0.5), (["Vintrans"], 0.5)],
            "PP": [(["P", "NP"], 1.0)],
            "Det": [(["the"], 0.5), (["a"], 0.5)],
            "Adj": [(["big"], 0.5), (["small"], 0.5)],
            "N": [(["cat"], 0.25), (["dog"], 0.25), (["telescope"], 0.25), (["park"], 0.25)],
            "Vtrans": [(["sees"], 0.5), (["likes"], 0.5)],
            "Vintrans": [(["sleeps"], 1.0)],
            "P": [(["with"], 0.5), (["in"], 0.5)]
        }
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
    tokenizer.add_special_tokens({
        #'pad_token': '[PAD]',
        'bos_token': '<|bos|>',
    })

    all_tokens = []
    for seq in sequences:
        text = ' '.join(seq)
        text_with_bos = tokenizer.bos_token + ' ' + text
        tokens = tokenizer.encode(text_with_bos)
        all_tokens.extend(tokens)

    # Save as binary token array
    os.makedirs(dataname, exist_ok=True)
    arr = np.array(all_tokens, dtype=np.uint32)
    arr.tofile(os.path.join(dataname, f'train.bin'))

    # Save metadata
    meta = {
        'vocab_size': tokenizer.vocab_size +1 ,
        'bos_token_id': tokenizer.bos_token_id,
        #'pad_token_id': tokenizer.pad_token_id
    }
    with open(os.path.join(dataname, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved {len(all_tokens)} tokens to {dataname}/train.bin")


pcfg = PCFG_basic()
generator = LanguageGenerator(pcfg)
sequences = generator.generate_sequences(DATASET_SIZE, MAX_SEQUENCE_LENGTH)
save_dataset(sequences, f'data/{pcfg.name}')