"""
Maze generation: random graphs with source s and target t.
"""
import random
from collections import defaultdict, deque


def bfs_reachable(adj, start):
    """Return the set of nodes reachable from start."""
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return visited


def ensure_connected(adj, n_nodes, rng):
    """
    Add edges to make the graph connected.
    Uses BFS to find connected components and links them with random edges.
    """
    # Find connected components
    visited = set()
    components = []
    for node in range(n_nodes):
        if node not in visited:
            comp = bfs_reachable(adj, node)
            # Only count nodes 0..n_nodes-1
            comp = {v for v in comp if v < n_nodes}
            components.append(comp)
            visited |= comp

    # Merge components by adding one bridge edge between consecutive components
    while len(components) > 1:
        u = rng.choice(list(components[0]))
        v = rng.choice(list(components[1]))
        adj[u].append(v)
        adj[v].append(u)
        merged = components[0] | components[1]
        components = [merged] + components[2:]


def generate_maze(n_nodes=20, edge_prob=0.15, seed=None):
    """
    Generate a random connected undirected graph.
    Returns (adj, s, t) where adj is a dict of node -> list[neighbor],
    s is the source, and t is the target (s != t, t reachable from s).
    """
    rng = random.Random(seed)

    adj = defaultdict(list)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                adj[i].append(j)
                adj[j].append(i)

    ensure_connected(adj, n_nodes, rng)

    # Pick source s uniformly at random
    s = rng.randint(0, n_nodes - 1)
    reachable = bfs_reachable(adj, s) - {s}

    # If graph has only one node (degenerate), add a self-loop-free node
    if not reachable:
        t = (s + 1) % n_nodes
        adj[s].append(t)
        adj[t].append(s)
    else:
        t = rng.choice(sorted(reachable))

    return dict(adj), s, t


def random_walk(adj, start, length, rng=None):
    """
    Perform a random walk of `length` steps starting from `start`.
    Returns a list of node IDs (including the starting node).
    Stays put if there are no neighbors (shouldn't happen on connected graphs).
    """
    if rng is None:
        rng = random
    walk = [start]
    cur = start
    for _ in range(length - 1):
        nbrs = adj.get(cur, [])
        if nbrs:
            cur = rng.choice(nbrs)
        walk.append(cur)
    return walk


def dfs_trace(adj, s, t, rng=None):
    """
    Run DFS from s searching for t.

    Returns (trace, path) where:
      trace : list of nodes visited in DFS order, including backtracking
              (when we backtrack to a node, we append it again to the trace).
      path  : list of nodes forming the s -> t path found by DFS,
              or None if t is unreachable.
    """
    if rng is None:
        rng = random

    visited = set()
    trace = []
    found_path = [None]

    def _dfs(node, current_path):
        visited.add(node)
        trace.append(node)

        if node == t:
            found_path[0] = list(current_path)
            return True

        neighbors = list(adj.get(node, []))
        rng.shuffle(neighbors)

        for nbr in neighbors:
            if nbr not in visited:
                current_path.append(nbr)
                if _dfs(nbr, current_path):
                    return True
                current_path.pop()
                # Backtrack: re-append the current node to mark return
                trace.append(node)

        return False

    _dfs(s, [s])
    return trace, found_path[0]


def bfs_shortest_path(adj, s, t):
    """Return the shortest path from s to t using BFS, or None if unreachable."""
    prev = {s: None}
    queue = deque([s])
    while queue:
        node = queue.popleft()
        if node == t:
            path = []
            cur = t
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path))
        for nbr in adj.get(node, []):
            if nbr not in prev:
                prev[nbr] = node
                queue.append(nbr)
    return None


def is_valid_path(adj, path, s, t):
    """Check that path is a valid s -> t walk (consecutive nodes are adjacent)."""
    if not path or path[0] != s or path[-1] != t:
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj.get(path[i], []):
            return False
    return True
