# write code that takes a rules of a pcfg and generates random sentences up to length n

import random

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

def main():
    rules = {
        'S': [(['NP', 'VP'], 0.9), (['VP'], 0.1)],
        'NP': [(['Det', 'N'], 1.0)],
        'VP': [(['V', 'NP'], 0.8), (['V'], 0.2)],
        'Det': [(['a'], 0.5), (['the'], 0.5)],
        'N': [(['cat'], 0.5), (['dog'], 0.5)],
        'V': [(['chases'], 1.0)]
    }

    pfcg = PCFG(rules)
    generator = LanguageGenerator(pfcg)
    sequences = generator.generate_sequences(10, 5)

    for seq in sequences:
        print(' '.join(seq))

if __name__ == "__main__":
    main()