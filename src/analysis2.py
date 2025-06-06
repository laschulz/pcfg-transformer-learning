from collections import defaultdict, Counter
from nltk import Tree, ViterbiParser, Nonterminal, Production, PCFG
import json
import argparse
from typing import List, Dict, Tuple, Any, Set, Optional

from generate_pcfg import PARSERS, dict_to_pcfg

def count_rules_in_tree(tree: Tree, rule_counts: dict):
    """
    Recursively count rule usage in a parse tree.
    """
    if isinstance(tree, Tree) and len(tree) > 0:
        lhs = tree.label()
        rhs = [child.label() if isinstance(child, Tree) else child for child in tree]
        rule = f"{lhs} -> {' '.join(rhs)}"
        rule_counts[rule] += 1
        for child in tree:
            if isinstance(child, Tree):
                count_rules_in_tree(child, rule_counts)

def to_cnf(parser: ViterbiParser) -> Tuple[Dict[str, list], Set[Nonterminal], Nonterminal]:
    """
    Convert an NLTK PCFG into CNF, but *also* keep track of unit‐productions.
    Returns:
      - cnf_rules: a dict with three keys: "unary_term", "unit", and "binary"
          • "unary_term": list of (A, terminal, prob) for rules A→"a"
          • "unit":       list of (A, B, prob) for rules A→B  (both Nonterminals)
          • "binary":     list of (A, B, C, prob) for rules A→B C
      - nonterminals: set of all Nonterminal symbols
      - start_symbol: the grammar’s start Nonterminal
    """
    grammar = parser.grammar()
    cnf_rules = {
        "unary": [],        # A → "a"  (terminal)
        "unit": [],         # A → B    (unit production between nonterminals)
        "binary": []        # A → B C
    }
    nonterminals: Set[Nonterminal] = set()
    start_symbol = grammar.start()

    # Step 1: Introduce temporary Nonterminals to replace terminals inside long RHS
    temp_rules: Dict[Any, Nonterminal] = {}
    next_temp_id = 0

    # First pass: Create temp_NTs for any terminal that appears in a binary/multi rule
    for production in grammar.productions():
        lhs = production.lhs()
        rhs = production.rhs()
        nonterminals.add(lhs)

        # If this RHS is length>1 and has a terminal, introduce a temp_NT
        if len(rhs) > 1:
            for symbol in rhs:
                if not isinstance(symbol, Nonterminal):
                    if symbol not in temp_rules:
                        temp_nt = Nonterminal(f"_TEMP{next_temp_id}")
                        next_temp_id += 1
                        temp_rules[symbol] = temp_nt
                        nonterminals.add(temp_nt)
                        # Add the “_TEMPx → terminal” as a unary‐term rule with prob = 1.0
                        cnf_rules["unary"].append((temp_nt, symbol, 1.0))

    # Second pass: turn every production into either unary, unit, or binary
    for production in grammar.productions():
        lhs = production.lhs()
        rhs = production.rhs()
        prob = production.prob()
        nonterminals.add(lhs)

        # CASE 1: A → terminal  (pure terminal rule)
        if len(rhs) == 1 and not isinstance(rhs[0], Nonterminal):
            cnf_rules["unary"].append((lhs, rhs[0], prob))

        # CASE 2: A → B   (unit production: both sides are Nonterminal)
        elif len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
            cnf_rules["unit"].append((lhs, rhs[0], prob))

        # CASE 3: A → B C   (both are Nonterminals, already CNF)
        elif len(rhs) == 2 and all(isinstance(sym, Nonterminal) for sym in rhs):
            cnf_rules["binary"].append((lhs, rhs[0], rhs[1], prob))

        # CASE 4: A → B a  or A → a B  (mixed terminal + nonterminal, length = 2)
        elif len(rhs) == 2:
            new_rhs = []
            for sym in rhs:
                if isinstance(sym, Nonterminal):
                    new_rhs.append(sym)
                else:
                    # replace terminal with its temp_NT
                    new_rhs.append(temp_rules[sym])
            cnf_rules["binary"].append((lhs, new_rhs[0], new_rhs[1], prob))

        # CASE 5: A → X1 X2 X3 ... (length > 2)
        else:  # len(rhs) > 2
            # First replace any terminal in RHS with temp_NT
            processed_rhs: List[Nonterminal] = []
            for sym in rhs:
                if isinstance(sym, Nonterminal):
                    processed_rhs.append(sym)
                else:
                    processed_rhs.append(temp_rules[sym])

            # Then binarize step by step
            current_lhs = lhs
            for i in range(len(processed_rhs) - 2):
                new_nt = Nonterminal(f"_BIN{next_temp_id}")
                next_temp_id += 1
                nonterminals.add(new_nt)

                # First binary: current_lhs → processed_rhs[i] new_nt
                # Use original prob only on the first introduced rule
                rule_prob = prob if i == 0 else 1.0
                cnf_rules["binary"].append(
                    (current_lhs, processed_rhs[i], new_nt, rule_prob)
                )
                current_lhs = new_nt

            # Final binary: current_lhs → second_last last
            last_two = processed_rhs[-2:]
            cnf_rules["binary"].append((current_lhs, last_two[0], last_two[1], 1.0))

    return cnf_rules, nonterminals, start_symbol

def cyk_parse(tokens: List[str], cnf_rules, nonterminals, start_symbol) -> Dict:
    """
    Run CYK algorithm on tokens with the given grammar.
    
    Returns:
      - valid: Whether the full sequence is valid
      - rule_counts: Counter of rules used in parse (or valid prefix)
      - valid_length: Length of valid prefix
      - invalid_suffix: Invalid part of the sequence
    """
    
    N = len(tokens)
    # Initialize charts
    chart = [[{} for _ in range(N+1)] for _ in range(N+1)]
    back = [[{} for _ in range(N+1)] for _ in range(N+1)]
    
    # Fill in base cases (length 1 spans)
    for i in range(N):
        t = tokens[i]
        for (A, terminal, prob) in cnf_rules["unary"]:
            if terminal == t:
                # record A → t with probability=prob
                if A not in chart[i][i+1] or prob > chart[i][i+1][A]:
                    chart[i][i+1][A] = prob
                    back[i][i+1][A] = ("term", terminal)

        # apply unit‐closure in cell [i, i+1]
        queue = list(chart[i][i+1].keys())
        while queue:
            B = queue.pop()
            pB = chart[i][i+1][B]
            for (A, B_rhs, unit_prob) in cnf_rules["unit"]:
                if B_rhs == B:
                    new_p = unit_prob * pB
                    if A not in chart[i][i+1] or new_p > chart[i][i+1][A]:
                        chart[i][i+1][A] = new_p
                        back[i][i+1][A] = ("unit", B)
                        queue.append(A)
    
    # Fill in the chart for longer spans
    for length in range(2, N+1):
        for start in range(N-length+1):
            end = start + length
            for split in range(start+1, end):
                left_cell = chart[start][split]
                right_cell = chart[split][end]
                if not left_cell or not right_cell:
                    continue
                # for each binary rule A → B C
                for (A, B, C, bprob) in cnf_rules["binary"]:
                    if B in left_cell and C in right_cell:
                        new_p = bprob * left_cell[B] * right_cell[C]
                        if A not in chart[start][end] or new_p > chart[start][end][A]:
                            chart[start][end][A] = new_p
                            back[start][end][A] = ("bin", split, B, C)

            # after placing all binary‐derived nonterminals in [start,end], do unit‐closure
            if chart[start][end]:
                queue = list(chart[start][end].keys())
                while queue:
                    X = queue.pop()
                    pX = chart[start][end][X]
                    for (A, B_rhs, unit_prob) in cnf_rules["unit"]:
                        if B_rhs == X:
                            new_p = unit_prob * pX
                            if A not in chart[start][end] or new_p > chart[start][end][A]:
                                chart[start][end][A] = new_p
                                back[start][end][A] = ("unit", X)
                                queue.append(A)
    # Find the longest valid prefix
    valid_length = 0
    for j in range(1, N+1):
        if start_symbol in chart[0][j]:
            valid_length = j
    
    # Extract rule counts from the chart
    rule_counts = defaultdict(int)
    
    def extract_rules(i, j, symbol):
        if i == j-1:  # Terminal case
            bp = back[i][j].get(symbol)
            if bp and bp[0] == "terminal":
                rule = f"{symbol} -> {bp[1]}"
                rule_counts[rule] += 1
        else:  # Non-terminal case
            bp = back[i][j].get(symbol)
            if bp and bp[0] == "binary":
                _, split, left_sym, right_sym = bp
                rule = f"{symbol} -> {left_sym} {right_sym}"
                rule_counts[rule] += 1
                extract_rules(i, split, left_sym)
                extract_rules(split, j, right_sym)
    
    if valid_length > 0:
        extract_rules(0, valid_length, start_symbol)
    
    # Check if full sequence is valid
    valid = (valid_length == N)
    invalid_suffix = tokens[valid_length:] if valid_length < N else []
    
    return {
        "valid": valid,
        "rule_counts": dict(rule_counts),
        "valid_length": valid_length,
        "invalid_suffix": invalid_suffix
    }

def analyze_sequences_enhanced(sequences, grammar):
    """
    Enhanced version that handles both valid and invalid sequences.
    
    Args:
      - sequences: List of token sequences
      - grammar: PCFG grammar
      
    Returns:
      - rule_counts: Default dict of rule frequencies
      - valid_sequences: Count of valid sequences
      - invalid_sequences: List of invalid sequences with their analysis
    """
    rule_counts = defaultdict(int)
    valid_count = 0
    invalid_analyses = []

    cnf_rules, nonterminals, start_symbol = to_cnf(grammar)
    print(cnf_rules)
    
    for seq in sequences:
        tokens = seq.split() if isinstance(seq, str) else seq
        
        # First try normal parsing for valid sequences
        try:
            parses = list(grammar.parse(tokens))
            if parses:
                # Valid sequence
                best_parse = parses[0]
                count_rules_in_tree(best_parse, rule_counts)
                valid_count += 1
                continue
        except ValueError:
            # If parsing fails, we'll handle it with CYK
            pass
            
        # For invalid sequences, use CYK to find valid prefix
        result = cyk_parse(tokens, cnf_rules, nonterminals, start_symbol)
        
        # Update global rule counts
        for rule, count in result["rule_counts"].items():
            rule_counts[rule] += count
            
        # Record invalid sequence analysis
        invalid_analyses.append({
            "sequence": tokens,
            "valid_prefix_length": result["valid_length"],
            "invalid_suffix": result["invalid_suffix"],
            "rules_in_prefix": result["rule_counts"]
        })
    
    return rule_counts, valid_count, invalid_analyses

def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze sequences with a PCFG")
    parser.add_argument("--grammar", type=str, required=True, help="Path to the PCFG grammar file")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch number for analysis")
    return parser.parse_args()

def main():
    args = argument_parser()
    pcfg = PARSERS[args.grammar]

    print(type(pcfg))

    results_path = "../results/results_log.json"
    with open(results_path, "r") as f:
        all_results = json.load(f)
    sequences = all_results[args.grammar][args.epoch]["generated_sequences"]

    # Use enhanced analysis function
    rule_counts, valid_count, invalid_analyses = analyze_sequences_enhanced(sequences, pcfg)
    
    print(f"Valid sequences: {valid_count}/{len(sequences)} ({valid_count/len(sequences)*100:.2f}%)")
    print("\nRule counts:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rule}: {count}")
    
    print(f"\nInvalid sequences: {len(invalid_analyses)}")
    if invalid_analyses:
        print("\nInvalid sequence analysis:")
        for i, analysis in enumerate(invalid_analyses[:10]):  # Show first 10 for brevity
            print(f"  Sequence {i+1}: Valid prefix length: {analysis['valid_prefix_length']}")
            print(f"    Invalid suffix: {' '.join(analysis['invalid_suffix'])}")

if __name__ == "__main__":
    main()