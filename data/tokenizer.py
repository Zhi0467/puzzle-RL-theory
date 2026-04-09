"""
Tokenizer for maze sequences.

Vocabulary layout:
  0 .. n_nodes-1  : node IDs
  n_nodes + 0     : SEP   — separates [graph | thinking | solution]
  n_nodes + 1     : EOS   — end of sequence
  n_nodes + 2     : PAD   — padding token (ignored in loss)
  n_nodes + 3     : EDGE  — marks start of neighbor list for a node
  n_nodes + 4     : S_TOK — marks the source node
  n_nodes + 5     : T_TOK — marks the target node

Sequence format:
  S_TOK [s] T_TOK [t]
  [0] EDGE [nbr ...] [1] EDGE [nbr ...] ...
  SEP
  [thinking tokens ...]
  SEP
  [solution tokens ...]
  EOS
"""


class MazeTokenizer:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.SEP   = n_nodes + 0
        self.EOS   = n_nodes + 1
        self.PAD   = n_nodes + 2
        self.EDGE  = n_nodes + 3
        self.S_TOK = n_nodes + 4
        self.T_TOK = n_nodes + 5
        self.vocab_size = n_nodes + 6

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_graph(self, adj: dict, s: int, t: int) -> list[int]:
        """Encode the graph as a flat token list (adjacency list format)."""
        tokens = [self.S_TOK, s, self.T_TOK, t]
        for node in range(self.n_nodes):
            tokens.append(node)
            tokens.append(self.EDGE)
            for nbr in sorted(adj.get(node, [])):
                tokens.append(nbr)
        return tokens

    def encode_sequence(
        self,
        adj: dict,
        s: int,
        t: int,
        thinking: list[int],
        solution: list[int],
    ) -> list[int]:
        """
        Build the full autoregressive sequence:
          [graph] SEP [thinking] SEP [solution] EOS
        """
        tokens = self.encode_graph(adj, s, t)
        tokens.append(self.SEP)
        tokens.extend(thinking)
        tokens.append(self.SEP)
        tokens.extend(solution)
        tokens.append(self.EOS)
        return tokens

    # ------------------------------------------------------------------
    # Decoding helpers (used during evaluation / generation)
    # ------------------------------------------------------------------

    def decode_generated(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        """
        Given a generated token sequence (after the graph+SEP prefix),
        split it into (thinking, solution) by locating the two SEP tokens.

        Returns (thinking_nodes, solution_nodes).  Node IDs outside
        [0, n_nodes) are filtered out from both parts.
        """
        sep_positions = [i for i, t in enumerate(tokens) if t == self.SEP]
        if len(sep_positions) < 2:
            return [], []

        think_start = sep_positions[0] + 1
        think_end   = sep_positions[1]
        sol_start   = sep_positions[1] + 1

        # Find EOS in solution part
        eos_positions = [
            i for i, tok in enumerate(tokens[sol_start:]) if tok == self.EOS
        ]
        sol_end = sol_start + eos_positions[0] if eos_positions else len(tokens)

        thinking = [t for t in tokens[think_start:think_end] if 0 <= t < self.n_nodes]
        solution = [t for t in tokens[sol_start:sol_end]   if 0 <= t < self.n_nodes]
        return thinking, solution
