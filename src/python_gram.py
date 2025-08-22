GRAMMARS = {
  "SafeMiniPythonPCFGv2": {

    "L0": [
      (["stmts_short"], 1.0)
    ],

    "stmts_short": [
      (["stmt"], 0.4),
      (["stmt", "stmt"], 0.4),
      (["stmt", "stmt", "stmt"], 0.2)
    ],

    "stmt": [
      (["small_stmt", "NL"], 0.6),
      (["compound_stmt"], 0.4)
    ],

    "small_stmt": [
      (["expr_stmt"], 0.55),
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
      (["PLUSEQ"], 0.34),
      (["MINUSEQ"], 0.33),
      (["STAREQ"], 0.33)
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
      (["if_stmt"], 0.45),
      (["for_stmt"], 0.25),
      (["WHILE", "test", "COLON", "suite"], 0.1),
      (["DEF", "NAME", "parameters", "COLON", "suite"], 0.2)
    ],

    "suite": [
      (["simple_stmt"], 0.7),
      (["NL", "INDENT", "block_1to2", "DEDENT"], 0.3)
    ],

    "block_1to2": [
      (["simple_stmt"], 0.55),
      (["simple_stmt", "simple_stmt"], 0.45)
    ],

    "if_stmt": [
      (["IF", "test", "COLON", "suite"], 0.25),
      (["IF", "test", "COLON", "suite", "ELSE", "COLON", "suite"], 0.35),
      (["IF", "test", "COLON", "suite", "ELIF", "test", "COLON", "suite"], 0.25),
      (["IF", "test", "COLON", "suite", "ELIF", "test", "COLON", "suite", "ELSE", "COLON", "suite"], 0.15)
    ],

    "for_stmt": [
      (["FOR", "NAME", "IN", "test", "COLON", "suite"], 1.0)
    ],

    "parameters": [
      (["LPAR", "RPAR"], 0.4),
      (["LPAR", "paramlist_short", "RPAR"], 0.6)
    ],

    "paramlist_short": [
      (["NAME"], 0.6),
      (["NAME", "COMMA", "NAME"], 0.4)
    ],

    "test": [
      (["and_test"], 0.75),
      (["and_test", "OR", "and_test"], 0.25)
    ], 

    "and_test": [
      (["not_test"], 0.8),
      (["not_test", "AND", "not_test"], 0.2)
    ],

    "not_test": [
      (["comparison"], 0.85),
      (["NOT", "comparison"], 0.15)
    ],

    "comparison": [
      (["arith_expr"], 0.75),
      (["arith_expr", "comp_op", "arith_expr"], 0.25)
    ],

    "comp_op": [
      (["EQEQ"], 0.3),
      (["NE"], 0.2),
      (["LT"], 0.2),
      (["GT"], 0.2),
      (["LE"], 0.05),
      (["GE"], 0.05)
    ],

    "arith_expr": [
      (["term"], 0.7),
      (["term", "PLUS", "term"], 0.15),
      (["term", "MINUS", "term"], 0.15)
    ],

    "term": [
      (["factor"], 0.7),
      (["factor", "STAR", "factor"], 0.2),
      (["factor", "SLASH", "factor"], 0.1)
    ],

    "factor": [
      (["atom_expr"], 0.85),
      (["PLUS", "atom_expr"], 0.05),
      (["MINUS", "atom_expr"], 0.05),
      (["TILDE", "atom_expr"], 0.05)
    ],

    "atom_expr": [
      (["atom"], 0.7),
      (["atom", "trailer"], 0.2),
      (["atom", "trailer", "trailer_opt"], 0.1)
    ],

    "trailer_opt": [
      ([], 0.6),
      (["trailer"], 0.4)
    ],

    "trailer": [
      (["LPAR", "RPAR"], 0.5),
      (["LPAR", "arglist_short", "RPAR"], 0.2),
      (["DOT", "NAME"], 0.2),
      (["LBRACK", "test", "RBRACK"], 0.1)
    ],

    "arglist_short": [
      (["test"], 0.7),
      (["test", "COMMA", "test"], 0.3)
    ],

    "atom": [
      (["NAME"], 0.35),
      (["NUMBER"], 0.25),
      (["STRING"], 0.2),
      (["list_lit"], 0.1),
      (["dict_lit"], 0.1)
    ],

    "list_lit": [
      (["LBRACK", "RBRACK"], 0.25),
      (["LBRACK", "test", "RBRACK"], 0.45),
      (["LBRACK", "test", "COMMA", "test", "RBRACK"], 0.3)
    ],

    "dict_lit": [
      (["LBRACE", "RBRACE"], 0.45),
      (["LBRACE", "test", "COLON", "test", "RBRACE"], 0.45),
      (["LBRACE", "test", "COLON", "test", "COMMA", "test", "COLON", "test", "RBRACE"], 0.1)
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
  }
}
