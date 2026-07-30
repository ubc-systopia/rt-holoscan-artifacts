import argparse
import random
import string

# Generates a random directed acyclic graph (DAG) in DOT format with:
# 1. Exactly one source node (first label) and one sink node (last label).
# 2. Node count, WCET bounds, extra-edge probability, graph name, and output file as arguments.
def generate_dag_dot(node_count, min_wcet, max_wcet, extra_edge_prob, graph_name, output):

    # Default non-layered DAG generation
    if node_count <= 26:
        labels = list(string.ascii_uppercase[:node_count])
    else:
        labels = [f"N{i}" for i in range(node_count)]

    wcets = {lbl: random.randint(min_wcet, max_wcet) for lbl in labels}

    edges = set()
    for i in range(node_count - 1):
        j = random.randint(i + 1, node_count - 1)
        edges.add((labels[i], labels[j]))

    for j in range(1, node_count):
        if not any(dst == labels[j] for _, dst in edges):
            i = random.randint(0, j - 1)
            edges.add((labels[i], labels[j]))

    for i in range(node_count - 1):
        for j in range(i + 1, node_count):
            if random.random() < extra_edge_prob:
                edges.add((labels[i], labels[j]))

    lines = [f"digraph {graph_name} {{"]
    for lbl in labels:
        lines.append(f"    {lbl} [WCET={wcets[lbl]}];")
    lines.append("")
    for src, dst in sorted(edges, key=lambda e: (labels.index(e[0]), labels.index(e[1]))):
        lines.append(f"    {src} -> {dst};")
    lines.append("}")

    with open(output, 'w') as f:
            f.write("\n".join(lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a random acyclic digraph (DAG) in DOT format')
    parser.add_argument('-n', '--nodes', type=int, default=9,
                        help='Total number of nodes in the graph')
    parser.add_argument('--min-wcet', type=int, default=100,
                        help='Minimum WCET value for nodes')
    parser.add_argument('--max-wcet', type=int, default=1000,
                        help='Maximum WCET value for nodes')
    parser.add_argument('-p', '--prob', type=float, default=0.2,
                        help='Probability of adding an extra edge (for each eligible pair)')
    parser.add_argument('-g', '--graph-name', type=str, default='G',
                        help='Name to use in the DOT `digraph` declaration')
    parser.add_argument('-o', '--output', type=str, default='out.dot',
                        help='Output DOT file path')
    parser.add_argument('--layered', action='store_true',
                        help='Generate graph with layered structure (tree-like levels)')
    args = parser.parse_args()

    dot_content = generate_dag_dot(
        node_count=args.nodes,
        min_wcet=args.min_wcet,
        max_wcet=args.max_wcet,
        extra_edge_prob=args.prob,
        graph_name=args.graph_name,
        layered=args.layered
    )

    with open(args.output, 'w') as f:
        f.write(dot_content)
    print(f"DOT file successfully written to {args.output}")
