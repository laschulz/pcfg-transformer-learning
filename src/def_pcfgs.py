GRAMMARS = {
  "PythonPCFG": {
    "STMTS": [
      (["stmt"],0.8),
       (["stmt", "STMTS"], 0.2),
    ],

    "stmt": [
      (["small_stmt", "NL"], 0.5), #0.5
      (["compound_stmt"], 0.5) #0.5
    ],

    "small_stmt": [
      (["small_stmt", "small_stmt"], 0.35),
      (["expr_stmt"], 0.2),
      (["return_stmt"], 0.15),
      (["import_stmt"], 0.15),
      (["PASS"], 0.05),
      (["BREAK"], 0.05),
      (["CONTINUE"], 0.05)
    ],

    "expr_stmt": [
      (["test"], 0.25),
      (["NAME", "EQ", "test"], 0.5),
      (["NAME", "augassign", "test"], 0.25)
    ],

    "augassign": [
      (["+="], 0.34),
      (["-="], 0.33),
      (["*="], 0.33)
    ],

    "return_stmt": [
      (["RETURN"], 0.3),
      (["RETURN", "test"], 0.7)
    ],

    "import_stmt": [
      (["IMPORT", "NAME"], 0.5),
      (["FROM", "NAME", "IMPORT", "NAME"], 0.5)
    ],

    "compound_stmt": [ #make this the subgrammar
      (["compound_stmt", "compound_stmt"], 0.2),
      (["compound_stmt_2"], 0.8)
    ],
    "compound_stmt_2": [
      (["if_stmt"], 0.35),
      (["FOR", "NAME", "IN", "test", ":", "suite"], 0.25),
      (["WHILE", "test", ":", "suite"], 0.2),
      (["DEF", "NAME", "parameters", ":", "suite"], 0.2)
    ], 
    "suite": [
      (["simple_stmt"], 0.9),
      (["INDENT", "compound_stmt_2", "DEDENT"], 0.1)
    ],
    "if_stmt": [
      (["IF", "test", ":", "suite"], 0.35),
      (["IF", "test", ":", "suite", "ELSE", ":", "suite"], 0.35),
      (["IF", "test", ":", "suite", "ELIF", "test", ":", "suite"], 0.15),
      (["IF", "test", ":", "suite", "ELIF", "test", ":", "suite", "ELSE", ":", "suite"], 0.15)
    ],

    "parameters": [
      (["(", ")"], 0.4),
      (["(", "PARAMS", ")"], 0.6)
    ],

    "PARAMS": [
      (["NAME"], 0.6),
      (["NAME", ",", "PARAMS"], 0.4)
    ],

    "test": [
      (["short_expr"], 0.8),
      (["short_expr", "binop", "short_expr"], 0.2)
    ],

    "binop": [
      (["+"], 0.34),
      (["-"], 0.33),
      (["*"], 0.18),
      (["/"], 0.15)
    ],

    "short_expr": [
      (["atom_expr"], 0.85),
      (["comparison_short"], 0.15)
    ],

    "comparison_short": [
      (["atom_expr", "comp_op", "atom_expr"], 1.0)
    ],

    "comp_op": [
      (["EQEQ"], 0.34),
      (["NE"], 0.22),
      (["LT"], 0.18),
      (["GT"], 0.18),
      (["LE"], 0.04),
      (["GE"], 0.04)
    ],

    "atom_expr": [
      (["atom"], 0.7),
      (["atom_expr", "trailer"], 0.3)
    ],

    "trailer": [
      (["(", ")"], 0.55),
      (["(", "test", ")"], 0.1),
      (["DOT", "NAME"], 0.25),
      (["(", "test", ")"], 0.1)
    ],

    "atom": [
      (["NAME"], 0.48),
      (["NUMBER"], 0.27),
      (["STRING"], 0.2),
      (["list_lit"], 0.03),
      (["dict_lit"], 0.02)
    ],

    "list_lit": [
      (["(", ")"], 0.88),
      (["(", "test", ")"], 0.12)
    ],

    "dict_lit": [
      (["{", "}"], 0.9),
      (["{", "test", ":", "test", "}"], 0.1)
    ],

    "NAME": [
      (["x"], 0.15), (["y"], 0.15), (["z"], 0.1),
      (["n"], 0.1), (["i"], 0.1), (["j"], 0.1),
      (["f"], 0.1), (["g"], 0.1), (["val"], 0.1)
    ],

    "NUMBER": [
      (["0"], 0.15), (["1"], 0.15), (["2"], 0.15), (["3"], 0.1),
      (["4"], 0.1), (["5"], 0.1), (["10"], 0.1), (["42"], 0.15)
    ],

    "STRING": [
      (["STR_A"], 0.4), (["STR_B"], 0.3), (["STR_HELLO"], 0.3)
    ]
  },

  "OverlappingSubgrammar": {
      "L0": [(["sL1", "L1", "eL1"], 0.5), (["sL1_2", "L1_2", "eL1_2"], 0.5)],
      "L1": [(["sL2", "L2", "eL2", "L1", "sL2_3", "L2_3", "eL2_3"], 0.4),
             (["sL2", "L2", "eL2", "L1"], 0.2), (["action"], 0.4)],
      "L1_2": [(["L1_2", "+", "sL2", "L2", "eL2"], 0.25),
               (["sL2", "L2", "eL2"], 0.75)],
      "L2": [(["cond"], 0.5), (["not", "L2"], 0.25),
             (["L2", "and", "L2"], 0.1), (["L2", "or", "L2"], 0.15)],
      "L2_3": [(["a", "L2_3"], 0.8), (["a"], 0.2)],
  },
  "KL_decomposition_example1": {
      "L1": [(["sL2_1", "L2_1", "eL2_1"], 0.3), 
              (["sL2_2", "L2_2", "eL2_2"], 0.3),
              (["sL2_3", "L2_3", "eL2_3"], 0.4)],
      "L2_1": [(["NUM"], 0.4), (["L2_1", "*", "L2_1"], 0.15), (["L2_1", "+", "L2_1"], 0.15), (["NUM", "NUM"], 0.3)],
      "NUM": [ (["0"], 0.2), (["1"], 0.2), (["2"], 0.2), (["3"], 0.2),
             (["4"], 0.1), (["5"], 0.1) ],
      "L2_2": [(["a", "L2_2", "b"], 0.6), (["c"], 0.4)],
      "L2_3": [(["x", "L2_3"], 0.8), (["x"], 0.2)],  
  }, 

  "KL_decomposition_example2": {
    "L1": [(["sL2_2", "L2_2", "eL2_2", "sL2_1", "L2_1", "eL2_1", "sL2_3", "L2_3", "eL2_3"], 1.0)],
    "L2_1": [(["NUM"], 0.4), (["L2_1", "*", "L2_1"], 0.15), (["L2_1", "+", "L2_1"], 0.15), (["NUM", "NUM"], 0.3)],
    "L2_2": [(["a", "L2_2", "b"], 0.6), (["c"], 0.4)],
    "L2_3": [(["x", "L2_3"], 0.8), (["x"], 0.2)],   
    "NUM": [ (["0"], 0.2), (["1"], 0.2), (["2"], 0.2), (["3"], 0.2),
             (["4"], 0.1), (["5"], 0.1) ]
  },      

  # subgrammars
  "L2_2": {
      "L2_2": [(["a", "L2_2", "b"], 0.6), (["c"], 0.4)],
  },
  "L2_1": {
     "L2_1": [(["NUM"], 0.4), (["L2_1", "*", "L2_1"], 0.15), (["L2_1", "+", "L2_1"], 0.15), (["NUM", "NUM"], 0.3)],
      "NUM": [ (["0"], 0.2), (["1"], 0.2), (["2"], 0.2), (["3"], 0.2),
             (["4"], 0.1), (["5"], 0.1)]
  },
  "L2_1_subgrammar": {
      "L1": [(["sL2_1", "L2_1", "eL2_1"], 1.0)],
      "L2_1": [(["NUM"], 0.4), (["L2_1", "*", "L2_1"], 0.15), (["L2_1", "+", "L2_1"], 0.15), (["NUM", "NUM"], 0.3)],
      "NUM": [ (["0"], 0.2), (["1"], 0.2), (["2"], 0.2), (["3"], 0.2),
             (["4"], 0.1), (["5"], 0.1)]
  },

  "L2_3": {
      "L2_3": [(["x", "L2_3"], 0.8), (["x"], 0.2)]  
  },

  "Grammar_with_simple_possibilities": {
      "START": [ (["sSUBJ", "SUBJ", "eSUBJ", "sVERB", "VERB", "eVERB", "sOBJ", "OBJ", "eOBJ"], 1.0)],
      "SUBJ": [ (["NOUN"], 0.2), (["a", "NOUN"], 0.4), (["the", "NOUN"], 0.4)],
      "NOUN": [ (["N"], 0.7), (["ADJ", "NOUN"], 0.3)],
      "VERB": [ (["V"], 0.3), (["V", "ADV"], 0.7)],
      "N": [ (["dog"], 0.2), (["cat"], 0.2), (["fox"], 0.1), 
            (["parrot"], 0.1), (["hamster"], 0.1), (["turtle"], 0.1),
            (["horse"], 0.1), (["pig"], 0.1)],
      "ADJ": [ (["big"], 0.2), (["poisonous"], 0.2), (["cute"], 0.2), (["lazy"], 0.2), (["quick"], 0.2)],
      "V": [ (["eats"], 0.15), (["runs"], 0.4), (["sleeps"], 0.15), 
            (["talks"], 0.15), (["cleans", "itself"], 0.15)],
      "ADV": [ (["quickly"], 0.2), (["slowly"], 0.3), (["happily"], 0.3), ("excitedly", 0.1), (["lazily"], 0.1)],
      "OBJ": [ (["blank"], 0.5), (["with", "SUBJ"], 0.5)]
  },

  "Grammar_with_only_simple_possibilities": {
      "START": [ (["sSUBJ", "SUBJ", "eSUBJ", "sVERB", "VERB", "eVERB", "sOBJ", "OBJ", "eOBJ"], 1.0)],
      "SUBJ": [ (["NOUN"], 1.0)],
      "NOUN": [ (["N"], 1.0)],
      "VERB": [ (["V"], 1.0)],
      "N": [ (["dog"], 0.2), (["cat"], 0.2), (["fox"], 0.1), 
          (["parrot"], 0.1), (["hamster"], 0.1), (["turtle"], 0.1),
          (["horse"], 0.1), (["pig"], 0.1)],
      "V": [ (["eats"], 0.15), (["runs"], 0.4), (["sleeps"], 0.15), 
          (["talks"], 0.15), (["cleans", "itself"], 0.15)],
      "OBJ": [ (["blank"], 1.0) ]
  },
  
  "Deeper_Recursion": {
      "L0": [ (["sL1", "L1", "eL1"], 0.7), (["L0", "L0"], 0.3) ],
      "L1": [ (["sL2", "L2", "eL2"], 0.6), (["L1", "L1"], 0.3), (["V"], 0.1) ],
      "L2": [  (["sL3", "L3", "eL3"], 0.6), (["L2", "L2"], 0.3), (["V"], 0.1) ],
      "L3": [  (["sL4", "L4", "eL4"], 0.6), (["L3", "L3"], 0.3), (["V"], 0.1) ],
      "L4": [  (["(", "V", ")"], 0.7), ("V", 0.3) ],
      "V": [ (["a"], 0.04), (["b"], 0.04), (["c"], 0.04), (["d"], 0.04), (["e"], 0.04),
                (["f"], 0.04), (["g"], 0.04), (["h"], 0.04), (["i"], 0.04), (["j"], 0.04),
                (["k"], 0.04), (["l"], 0.04), (["m"], 0.04), (["n"], 0.04), (["o"], 0.04),
                (["p"], 0.04), (["q"], 0.04), (["r"], 0.04), (["s"], 0.04), (["t"], 0.04),
                (["u"], 0.04), (["v"], 0.04), (["w"], 0.04), (["x"], 0.04), (["y"], 0.04)]
  },
  # subgrammars
  "L1": {
      "L1": [ (["sL2", "L2", "eL2"], 0.6), (["L1", "L1"], 0.3), (["V"], 0.1) ],
      "L2": [  (["sL3", "L3", "eL3"], 0.6), (["L2", "L2"], 0.3), (["V"], 0.1) ],
      "L3": [  (["sL4", "L4", "eL4"], 0.6), (["L3", "L3"],  0.3), (["V"], 0.1) ],
      "L4": [  (["(", "V", ")"], 0.7), ("V", 0.3) ],
      "V": [ (["a"], 0.04), (["b"], 0.04), (["c"], 0.04), (["d"], 0.04), (["e"], 0.04),
                (["f"], 0.04), (["g"], 0.04), (["h"], 0.04), (["i"], 0.04), (["j"], 0.04),
                (["k"], 0.04), (["l"], 0.04), (["m"], 0.04), (["n"], 0.04), (["o"], 0.04),
                (["p"], 0.04), (["q"], 0.04), (["r"], 0.04), (["s"], 0.04), (["t"], 0.04),
                (["u"], 0.04), (["v"], 0.04), (["w"], 0.04), (["x"], 0.04), (["y"], 0.04)]
  },

  "L2": {
    "L2": [  (["sL3", "L3", "eL3"], 0.6), (["L2", "L2"], 0.3), (["V"], 0.1) ],
      "L3": [  (["sL4", "L4", "eL4"], 0.6), (["L3", "L3"], 0.3), (["V"], 0.1) ],
      "L4": [  (["(", "V", ")"], 0.7), ("V", 0.3) ],
      "V": [ (["a"], 0.04), (["b"], 0.04), (["c"], 0.04), (["d"], 0.04), (["e"], 0.04),
                (["f"], 0.04), (["g"], 0.04), (["h"], 0.04), (["i"], 0.04), (["j"], 0.04),
                (["k"], 0.04), (["l"], 0.04), (["m"], 0.04), (["n"], 0.04), (["o"], 0.04),
                (["p"], 0.04), (["q"], 0.04), (["r"], 0.04), (["s"], 0.04), (["t"], 0.04),
                (["u"], 0.04), (["v"], 0.04), (["w"], 0.04), (["x"], 0.04), (["y"], 0.04)]
  },

  "L3": {
      "L3": [  (["sL4", "L4", "eL4"], 0.6), (["L3", "L3"], 0.3), (["V"], 0.1) ],
      "L4": [  (["(", "V", ")"], 0.7), ("V", 0.3) ],
      "V": [ (["a"], 0.04), (["b"], 0.04), (["c"], 0.04), (["d"], 0.04), (["e"], 0.04),
                (["f"], 0.04), (["g"], 0.04), (["h"], 0.04), (["i"], 0.04), (["j"], 0.04),
                (["k"], 0.04), (["l"], 0.04), (["m"], 0.04), (["n"], 0.04), (["o"], 0.04),
                (["p"], 0.04), (["q"], 0.04), (["r"], 0.04), (["s"], 0.04), (["t"], 0.04),
                (["u"], 0.04), (["v"], 0.04), (["w"], 0.04), (["x"], 0.04), (["y"], 0.04)]
  },

  "L4": {
      "L4": [ (["(", "V", ")"], 0.7), ("V", 0.3) ],
      "V": [ (["a"], 0.04), (["b"], 0.04), (["c"], 0.04), (["d"], 0.04), (["e"], 0.04),
                (["f"], 0.04), (["g"], 0.04), (["h"], 0.04), (["i"], 0.04), (["j"], 0.04),
                (["k"], 0.04), (["l"], 0.04), (["m"], 0.04), (["n"], 0.04), (["o"], 0.04),
                (["p"], 0.04), (["q"], 0.04), (["r"], 0.04), (["s"], 0.04), (["t"], 0.04),
                (["u"], 0.04), (["v"], 0.04), (["w"], 0.04), (["x"], 0.04), (["y"], 0.04)]
  },

  "nestedParentheses": {
    "L4": [ (["(", "L5", ")"], 0.8), (["L4", "L4"], 0.2) ],
    "L5": [ (["(", "L5", ")"], 0.8), ("a", 0.2)],
  },

  "ABC_grammar": {
      "L0": [(["sL1a", "L1a", "eL1a", "sL1b", "L1b", "eL1b", "sL1c", "L1c", "eL1c"], 1.0)],
      "L1a": [(["sL2", "L2", "eL2", "L1a", "sL2_3", "L2_3", "eL2_3"], 0.4),
              (["sL2", "L2", "eL2", "L1a"], 0.2), (["action"], 0.4)],
      "L2": [(["sL4", "L4", "eL4"], 0.5), (["not", "L2"], 0.25),
              (["L2", "and", "L2"], 0.1), (["L2", "or", "L2"], 0.15)],
      "L2_3": [(["a", "L2_3"], 0.8), (["a"], 0.2)],
      "L4": [(["=="], 0.2), (["<="], 0.2), (["<"], 0.2),
              ([">="], 0.2), ([">"], 0.2)],  
      "L1b": [(["L1b", "+", "sL2_2", "L2_2", "eL2_2"], 0.25),
                (["sL2_2", "L2_2", "eL2_2"], 0.75)],
      "L2_2": [(["a", "L2_2", "b"], 0.6), (["c"], 0.4)],
      "L1c": [(["xy", "L1c"], 0.3), (["x", "L1c"], 0.3),
                (["sL2_3c", "L2_3c", "eL2_3c"], 0.4)],
      "L2_3c": [(["c", "L2_3c"], 0.7), (["c"], 0.3)]
  }, 

  "L1b": {
    "L1b": [(["L1b", "+", "sL2_2", "L2_2", "eL2_2"], 0.25),
      (["sL2_2", "L2_2", "eL2_2"], 0.75)],     
    "L2_2": [(["a", "L2_2", "b"], 0.6), (["c"], 0.4)],
  },
  "L1b_subgrammar": {
    "L0": [(["sL1b", "L1b", "eL1b"], 1.0)],
    "L1b": [(["L1b", "+", "sL2_2", "L2_2", "eL2_2"], 0.25),
      (["sL2_2", "L2_2", "eL2_2"], 0.75)],     
    "L2_2": [(["a", "L2_2", "b"], 0.6), (["c"], 0.4)],
  },
  "L1c": {
    "L1c": [(["xy", "L1c"], 0.3), (["x", "L1c"], 0.3),
            (["sL2_3c", "L2_3c", "eL2_3c"], 0.4)],
    "L2_3c": [(["c", "L2_3c"], 0.7), (["c"], 0.3)]
  },
  "L1c_subgrammar": {
    "L0": [(["sL1c", "L1c", "eL1c"], 1.0)],
    "L1c": [(["xy", "L1c"], 0.3), (["x", "L1c"], 0.3),
            (["sL2_3c", "L2_3c", "eL2_3c"], 0.4)],
    "L2_3c": [(["c", "L2_3c"], 0.7), (["c"], 0.3)]
  },

   # "TriLingo": {

  #   # -------------------------
  #   # Top-level selector
  #   # -------------------------
  #   "L0": [
  #     (["ARITH"], 0.34),
  #     (["PYEXPR"], 0.33),
  #     (["ENG"], 0.33)
  #   ],

  #   # =========================
  #   # 1) Arithmetic subgrammar
  #   # =========================

  #   # Infix chain: E -> E op T | T
  #   "ARITH": [
  #     (["A_TERM"], 0.65),                        # stop
  #     (["ARITH", "A_ADDOP", "A_TERM"], 0.35)    # continue (E ops ≈ 0.54)
  #   ],
  #   # T -> T * F | F
  #   "A_TERM": [
  #     (["A_FACTOR"], 0.7),
  #     (["A_TERM", "A_MULOP", "A_FACTOR"], 0.3)
  #   ],
  #   # F -> number | (E) | -F | F ^ F   (power recursion light)
  #   "A_FACTOR": [
  #     (["NUMBER"], 0.4),
  #     (["VAR"], 0.2),
  #     (["(", "ARITH", ")"], 0.1),
  #     (["-", "A_FACTOR"], 0.15),
  #     (["A_FACTOR", "^", "A_FACTOR"], 0.15)
  #   ],

  #   "A_ADDOP": [
  #     (["+"], 0.5), (["-"], 0.5)
  #   ],
  #   "A_MULOP": [
  #     (["*"], 0.6), (["/"], 0.4)
  #   ],
  #   "NUMBER": [
  #     (["0"], 0.2), (["1"], 0.2), (["2"], 0.2), (["3"], 0.2),
  #     (["4"], 0.1), (["5"], 0.1)
  #   ],
  #       # Token leaves for python-ish literals (keep tiny for stability)
  #   "VAR": [   # (redeclared above for ENG; if you need one unified NAME, keep this small set)
  #     (["x"], 0.22), (["y"], 0.18), (["z"], 0.25),
  #     (["f"], 0.15), (["g"], 0.2)
  #   ],

  #   # =========================
  #   # 2) Python-expression subgrammar
  #   # =========================

  #   # Infix chain: E -> E op Atom | Atom
  #   "PYEXPR": [
  #     (["STMTS"], 1.0),
  #   ],
  #   "STMTS": [
  #     (["stmt"],0.8),
  #      (["stmt", "STMTS"], 0.2),
  #   ],

  #   "stmt": [
  #     (["small_stmt", "NL"], 0.3),
  #     (["compound_stmt"], 0.7)
  #   ],

  #   "small_stmt": [
  #     (["small_stmt", "small_stmt"], 0.35),
  #     (["expr_stmt"], 0.2),
  #     (["return_stmt"], 0.15),
  #     (["import_stmt"], 0.15),
  #     (["PASS"], 0.05),
  #     (["BREAK"], 0.05),
  #     (["CONTINUE"], 0.05)
  #   ],

  #   "expr_stmt": [
  #     (["test"], 0.25),
  #     (["NAME", "EQ", "test"], 0.5),
  #     (["NAME", "augassign", "test"], 0.25)
  #   ],

  #   "augassign": [
  #     (["+="], 0.34),
  #     (["-="], 0.33),
  #     (["*="], 0.33)
  #   ],

  #   "return_stmt": [
  #     (["RETURN"], 0.3),
  #     (["RETURN", "test"], 0.7)
  #   ],

  #   "import_stmt": [
  #     (["IMPORT", "NAME"], 0.5),
  #     (["FROM", "NAME", "IMPORT", "NAME"], 0.5)
  #   ],

  #   "compound_stmt": [ #make this the subgrammar
  #     (["compound_stmt", "compound_stmt"], 0.2),
  #     (["compound_stmt_2"], 0.8)
  #   ],
  #   "compound_stmt_2": [
  #     (["if_stmt"], 0.35),
  #     (["FOR", "NAME", "IN", "test", ":", "suite"], 0.25),
  #     (["WHILE", "test", ":", "suite"], 0.2),
  #     (["DEF", "NAME", "parameters", ":", "suite"], 0.2)
  #   ], 
  #   "suite": [
  #     (["simple_stmt"], 0.9),
  #     (["INDENT", "compound_stmt_2", "DEDENT"], 0.1)
  #   ],
  #   "if_stmt": [
  #     (["IF", "test", ":", "suite"], 0.35),
  #     (["IF", "test", ":", "suite", "ELSE", ":", "suite"], 0.35),
  #     (["IF", "test", ":", "suite", "ELIF", "test", ":", "suite"], 0.15),
  #     (["IF", "test", ":", "suite", "ELIF", "test", ":", "suite", "ELSE", ":", "suite"], 0.15)
  #   ],

  #   "parameters": [
  #     (["(", ")"], 0.4),
  #     (["(", "PARAMS", ")"], 0.6)
  #   ],

  #   "PARAMS": [
  #     (["NAME"], 0.6),
  #     (["NAME", ",", "PARAMS"], 0.4)
  #   ],

  #   "test": [
  #     (["short_expr"], 0.8),
  #     (["short_expr", "binop", "short_expr"], 0.2)
  #   ],

  #   "binop": [
  #     (["+"], 0.34),
  #     (["-"], 0.33),
  #     (["*"], 0.18),
  #     (["/"], 0.15)
  #   ],

  #   "short_expr": [
  #     (["atom_expr"], 0.85),
  #     (["comparison_short"], 0.15)
  #   ],

  #   "comparison_short": [
  #     (["atom_expr", "comp_op", "atom_expr"], 1.0)
  #   ],

  #   "comp_op": [
  #     (["=="], 0.34),
  #     (["!="], 0.22),
  #     (["<"], 0.18),
  #     ([">"], 0.18),
  #     (["<="], 0.04),
  #     ([">="], 0.04)
  #   ],

  #   "atom_expr": [
  #     (["atom"], 0.7),
  #     (["atom_expr", "trailer"], 0.3)
  #   ],

  #   "trailer": [
  #     (["(", ")"], 0.55),
  #     (["(", "test", ")"], 0.1),
  #     ([".", "NAME"], 0.25),
  #     (["(", "test", ")"], 0.1)
  #   ],

  #   "atom": [
  #     (["NUMBER"], 0.4),
  #     (["VAR"], 0.4),
  #     (["list_lit"], 0.1),
  #     (["dict_lit"], 0.1)
  #   ],

  #   "list_lit": [
  #     (["(", ")"], 0.88),
  #     (["(", "test", ")"], 0.12)
  #   ],

  #   "dict_lit": [
  #     (["{", "}"], 0.9),
  #     (["{", "test", ":", "test", "}"], 0.1)
  #   ],

  #   # =========================
  #   # 3) English subgrammar
  #   # =========================

  #   # Coordination recursion: S -> S CONJ S | NP VP
  #   "ENG": [
  #     (["CLAUSE"], 0.7),
  #     (["ENG", "CONJ", "ENG"], 0.3)      # recursive coordination
  #   ],

  #   "CLAUSE": [
  #     (["NP", "VP"], 0.8),
  #     (["PP", "NP", "VP"], 0.1),
  #     (["AdvP", "NP", "VP"], 0.1)
  #   ],

  #   # NP with recursive PP-attachment and names
  #   "NP": [
  #     (["NPRIME"], 0.75),
  #     (["NP", "CONJ", "NP"], 0.25)           # NP coordination (recursive)
  #   ],

  #   # N' (N-bar) carries modifiers: adjectives, PP, relative clauses
  #   "NPRIME": [
  #     (["DET", "NBAR"], 0.6),
  #     (["NAME"], 0.2),
  #     (["PRON"], 0.2)
  #   ],

  #   # Core N-bar expansion with layering
  #   "NBAR": [
  #     (["N"], 0.35),                         # bare noun
  #     (["AP", "NBAR"], 0.25),                # adjectival stack
  #     (["N", "PP"], 0.15),                   # post-nominal PP
  #     (["N", "RELCL"], 0.15),                # relative clause
  #     (["N", "PP", "RELCL"], 0.10)           # both
  #   ],

  #   # Relative clause: that/who/which + VP (subject relative)
  #   "RELCL": [
  #     (["RELPRO", "VP"], 1.0)
  #   ],

  #   # Adjective phrase: right-recursive stack
  #   "AP": [
  #     (["ADJ"], 0.6),
  #     (["ADJ", "AP"], 0.4)
  #   ],

  #   # ---------- Verb Phrases with complements & modifiers ----------
  #   "VP": [
  #     (["V"], 0.1),                          # intransitive
  #     (["V", "NP"], 0.35),                   # transitive
  #     (["V", "PP"], 0.1),                    # PP complement/adjunct
  #     (["V", "NP", "PP"], 0.15),
  #     (["V", "CP"], 0.1),                    # sentential complement (that S, because S, if S)
  #     (["AUX", "VP"], 0.1),                  # auxiliary chain (recursive)
  #     (["VP", "PP"], 0.1)                    # VP + PP (adjunct stacking, mild recursion)
  #   ],

  #   # Sentential complement: COMP S
  #   "CP": [
  #     (["COMP", "S"], 1.0)
  #   ],

  #   # Adverb phrase (can sit before a clause in CLAUSE, or use VP->VP AdvP if you want more)
  #   "AdvP": [
  #     (["ADV"], 0.65),
  #     (["ADV", "AdvP"], 0.35)
  #   ],

  #   # ---------- Prepositional Phrases (already recursive via NP) ----------
  #   "PP": [
  #     (["P", "NP"], 0.7),
  #     (["P", "NP", "PP"], 0.3)               # right recursion PP tail
  #   ],

  #   # ---------- Lexicons (kept small; feel free to merge with your existing sets) ----------
  #   "DET": [
  #     (["the"], 0.6), (["a"], 0.4)
  #   ],
  #   "N": [
  #     (["dog"], 0.2), (["cat"], 0.2), (["person"], 0.2),
  #     (["robot"], 0.2), (["hamster"], 0.2)
  #   ],
  #   "V": [
  #     (["sees"], 0.15), (["likes"], 0.15), (["runs"], 0.15),
  #     (["chases"], 0.15), (["computes"], 0.15),
  #     (["thinks"], 0.15), (["says"], 0.10)          # take CP complements naturally
  #   ],
  #   "P": [
  #     (["with"], 0.35), (["in"], 0.3), (["on"], 0.2), (["under"], 0.15)
  #   ],
  #   "CONJ": [
  #     (["and"], 0.65), (["or"], 0.35)
  #   ],
  #   "NAME": [
  #     (["Alice"], 0.3), (["Bob"], 0.3), (["Eve"], 0.4)
  #   ],
  #   "PRON": [
  #     (["she"], 0.34), (["he"], 0.33), (["they"], 0.33)
  #   ],
  #   "ADJ": [
  #     (["big"], 0.25), (["small"], 0.25), (["fast"], 0.25), (["curious"], 0.25)
  #   ],
  #   "ADV": [
  #     (["quickly"], 0.3), (["slowly"], 0.3), (["quietly"], 0.4)
  #   ],
  #   "AUX": [
  #     (["will"], 0.5), (["can"], 0.5)
  #   ],
  #   "COMP": [
  #     (["that"], 0.5), (["because"], 0.3), (["if"], 0.2)
  #   ],
  #   "RELPRO": [
  #     (["that"], 0.5), (["who"], 0.25), (["which"], 0.25)
  #   ],
  # },

 # "PythonPCFG_symbol": {
  #   "STMTS": [
  #     (["stmt"],0.8),
  #      (["stmt", "STMTS"], 0.2),
  #   ],

  #   "stmt": [
  #     (["small_stmt", "NL"], 0.5), #0.5
  #     (["compound_stmt"], 0.5) #0.5
  #   ],

  #   "small_stmt": [
  #     (["small_stmt", "small_stmt"], 0.35),
  #     (["expr_stmt"], 0.2),
  #     (["return_stmt"], 0.15),
  #     (["import_stmt"], 0.15),
  #     (["PASS"], 0.05),
  #     (["BREAK"], 0.05),
  #     (["CONTINUE"], 0.05)
  #   ],

  #   "expr_stmt": [
  #     (["test"], 0.25),
  #     (["NAME", "EQ", "test"], 0.5),
  #     (["NAME", "augassign", "test"], 0.25)
  #   ],

  #   "augassign": [
  #     (["+="], 0.34),
  #     (["-="], 0.33),
  #     (["*="], 0.33)
  #   ],

  #   "return_stmt": [
  #     (["RETURN"], 0.3),
  #     (["RETURN", "test"], 0.7)
  #   ],

  #   "import_stmt": [
  #     (["IMPORT", "NAME"], 0.5),
  #     (["FROM", "NAME", "IMPORT", "NAME"], 0.5)
  #   ],

  #   "suite": [
  #     (["simple_stmt"], 0.9),
  #     (["INDENT", "compound_stmt_2", "DEDENT"], 0.1)
  #   ],
  #   "if_stmt": [
  #     (["IF", "test", ":", "suite"], 0.35),
  #     (["IF", "test", ":", "suite", "ELSE", ":", "suite"], 0.35),
  #     (["IF", "test", ":", "suite", "ELIF", "test", ":", "suite"], 0.15),
  #     (["IF", "test", ":", "suite", "ELIF", "test", ":", "suite", "ELSE", ":", "suite"], 0.15)
  #   ],

  #   "parameters": [
  #     (["(", ")"], 0.4),
  #     (["(", "PARAMS", ")"], 0.6)
  #   ],

  #   "PARAMS": [
  #     (["NAME"], 0.6),
  #     (["NAME", ",", "PARAMS"], 0.4)
  #   ],

  #   "test": [
  #     (["short_expr"], 0.8),
  #     (["short_expr", "binop", "short_expr"], 0.2)
  #   ],

  #   "binop": [
  #     (["+"], 0.34),
  #     (["-"], 0.33),
  #     (["*"], 0.18),
  #     (["/"], 0.15)
  #   ],

  #   "short_expr": [
  #     (["atom_expr"], 0.85),
  #     (["comparison_short"], 0.15)
  #   ],

  #   "comparison_short": [
  #     (["atom_expr", "comp_op", "atom_expr"], 1.0)
  #   ],

  #   "comp_op": [
  #     (["EQEQ"], 0.34),
  #     (["NE"], 0.22),
  #     (["LT"], 0.18),
  #     (["GT"], 0.18),
  #     (["LE"], 0.04),
  #     (["GE"], 0.04)
  #   ],

  #   "atom_expr": [
  #     (["atom"], 0.7),
  #     (["atom_expr", "trailer"], 0.3)
  #   ],

  #   "trailer": [
  #     (["(", ")"], 0.55),
  #     (["(", "test", ")"], 0.1),
  #     (["DOT", "NAME"], 0.25),
  #     (["(", "test", ")"], 0.1)
  #   ],

  #   "atom": [
  #     (["NAME"], 0.48),
  #     (["NUMBER"], 0.27),
  #     (["STRING"], 0.2),
  #     (["list_lit"], 0.03),
  #     (["dict_lit"], 0.02)
  #   ],

  #   "list_lit": [
  #     (["(", ")"], 0.88),
  #     (["(", "test", ")"], 0.12)
  #   ],

  #   "dict_lit": [
  #     (["{", "}"], 0.9),
  #     (["{", "test", ":", "test", "}"], 0.1)
  #   ],

  #   "NAME": [
  #     (["x"], 0.15), (["y"], 0.15), (["z"], 0.1),
  #     (["n"], 0.1), (["i"], 0.1), (["j"], 0.1),
  #     (["f"], 0.1), (["g"], 0.1), (["val"], 0.1)
  #   ],

  #   "NUMBER": [
  #     (["0"], 0.15), (["1"], 0.15), (["2"], 0.15), (["3"], 0.1),
  #     (["4"], 0.1), (["5"], 0.1), (["10"], 0.1), (["42"], 0.15)
  #   ],

  #   "STRING": [
  #     (["STR_A"], 0.4), (["STR_B"], 0.3), (["STR_HELLO"], 0.3)
  #   ]
  # },
}