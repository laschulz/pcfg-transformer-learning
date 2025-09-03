from analysis_hierarchy import analyze_hieararchy_all_epochs
import argparse
import json

SEEDS = [2, 3]
#path = '../data/PythonPCFG/PythonPCFG_100000_STMTS/OneLayer/new/'

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grammar', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nonTerminal', type=str, required=True)
    parser.add_argument('--subgrammar', type=str, required=True)
    parser.add_argument('--to_epoch', type=int, required=True)
    parser.add_argument('--train_type', type=str, required=True)

    return parser.parse_args()

def main():
    args = argument_parser()
    x = []
    # for s in SEEDS:
    #     all_grammar_results = analyze_hieararchy_all_epochs(args.grammar, args.nonTerminal, args.subgrammar, args.to_epoch, args.dataset_name, args.model, args.train_type, seed=s)
    #     x.append(all_grammar_results)
    
    # # # save x in json file
    master_results_path = "../results/hierarchy_analysis_x.json"
    # with open(master_results_path, 'w') as f:
    #     json.dump(x, f, indent=4)
    
    with open(master_results_path, 'r') as f:
        x = json.load(f)
    all_grammar_results = x[0]
    for ckpt in all_grammar_results[args.model][args.grammar]["2"][args.nonTerminal]:
        all_grammar_results[args.model][args.grammar]["2"][args.nonTerminal][ckpt]['kl_divergence'] = sum([x[i][args.model][args.grammar][f'{SEEDS[i]}'][args.nonTerminal][ckpt]['kl_divergence'] for i in range(len(SEEDS))]) / len(SEEDS)
    
    master_results_path = "../results/hierarchy_analysis.json"
    with open(master_results_path, 'w') as f:
        json.dump(all_grammar_results, f, indent=4)

if __name__ == "__main__":
    main()