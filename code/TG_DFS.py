import itertools
import re
import time
import uuid
from collections import defaultdict

import networkx as nx

def read_dot_file(dot_file):
    graph = nx.nx_pydot.read_dot(dot_file)

    G = nx.DiGraph()

    for node, data in graph.nodes(data=True):
        G.add_node(node, **data)
    
    for edge in graph.edges(data=True):
        u, v, data = edge
        G.add_edge(u, v, **data)

    return G

# For the path specific value
class SuccessorChecker:
    def __init__(self, graph, level_map, queuesize):
        self.queuesize = queuesize  

        # Map each node to its minimum level * queuesize
        self.node_to_level = {}
        for level, nodes in level_map.items():
            for node in nodes:
                adjusted_level = level * self.queuesize
                if node not in self.node_to_level:
                    self.node_to_level[node] = adjusted_level
                else:
                    self.node_to_level[node] = min(self.node_to_level[node], adjusted_level)

        # Precompute all transitive successors
        self.all_successors = {
            node: nx.descendants(graph, node) for node in graph.nodes
        }

    # Return True if the node's minimum level is strictly lower than the given level
    def is_strictly_lower_level(self, node, level):
        node_level = self.node_to_level.get(node)
        return node_level is not None and node_level < level

    # Return True if the node's minimum level is exactly equal to the given level
    def is_at_level(self, node, level):
        return self.node_to_level.get(node) == level

    # Check if any transitive successor is at a suitable level
    def has_successor_in_lower_or_equal_level(self, node, level):
        for succ in self.all_successors.get(node, set()):
            succ_level = self.node_to_level.get(succ)
            if succ_level is not None and succ_level <= level:
                return True
        return False

def dfs_post_order(tree, queuesize, node, node_connections, iteration, memo, termination_set, checker, sink):

    attributes = tree.nodes[node]

    # Check if self has been memoized
    if attributes["graph_node"]+"_"+str(attributes["iteration"])+str(attributes["start"]) in memo:
        attributes["value"] = memo[attributes["graph_node"]+"_"+str(attributes["iteration"])+str(attributes["start"])]
        return

    # Create next level of tree if we are a starting node, else add start
    if attributes["start"]:

        # Autoconcurrency edges
        if (attributes["graph_node"] == sink) or (queuesize > 1):
            id = uuid.uuid4()
            tree.add_node(id, graph_node=attributes["graph_node"], iteration=iteration+1, value="", weight=attributes["graph_node"]+"_"+str(iteration+1), start=False)  
            tree.add_edge(node, id)

        # Normal edges
        for pred in node_connections[attributes["graph_node"]]["predecessors"]:
            id = uuid.uuid4()
            tree.add_node(id, graph_node=pred, iteration=iteration, value="", weight=pred+"_"+str(iteration), start=False)  
            tree.add_edge(node, id)

        # Downstream edges
        for succ in node_connections[attributes["graph_node"]]["successors"]:
            id = uuid.uuid4()
            tree.add_node(id, graph_node=succ, iteration=iteration+queuesize, value="", weight="", start=True)  
            tree.add_edge(node, id)

    else:
        id = uuid.uuid4()
        tree.add_node(id, graph_node=attributes["graph_node"], iteration=iteration, value="", weight="", start=True)  
        tree.add_edge(node, id)

    # For each child, either terminate or continue recursion
    for child in tree.successors(node):

        childattributes = tree.nodes[child]

        childname = childattributes["graph_node"]+"_"+str(childattributes["iteration"])+str(childattributes["start"])
        
        # Termination check
        if childattributes["start"] == True:
        
            #I have a 0 cost path to source
            if checker.is_at_level(childattributes["graph_node"], childattributes["iteration"]):
                #But I also have a path to someone else with a 0 cost path to source
                if checker.has_successor_in_lower_or_equal_level(childattributes["graph_node"], childattributes["iteration"]):
                    tree.nodes[child]["value"] = childname+"!"       
                else:
                    tree.nodes[child]["value"] = childname           

            #I have a path to someone with a 0 cost path to source
            elif checker.has_successor_in_lower_or_equal_level(childattributes["graph_node"], childattributes["iteration"]):
                tree.nodes[child]["value"] = childname+"!"

            #I have a path to another version of me which has a 0 cost path to source
            elif checker.is_strictly_lower_level(childattributes["graph_node"], childattributes["iteration"]):
                tree.nodes[child]["value"] = childname+"!"

            else:
                dfs_post_order(tree, queuesize, child, node_connections, childattributes["iteration"], memo, termination_set, checker, sink)

        else:
            dfs_post_order(tree, queuesize, child, node_connections, childattributes["iteration"], memo, termination_set, checker, sink)


    # Value is array of all child values, plus own weight
    childvalues = [tree.nodes[child]["value"] for child in tree.successors(node)]
    
    flattened = [item for sublist in childvalues for item in (sublist if isinstance(sublist, list) else [sublist])]

    if childvalues == ['']:
        attributes["value"] = attributes["weight"]
    else:
        flattened = list(filter(None, flattened))
        if attributes["weight"] == '': 
            attributes["value"] = [attributes["weight"] + item for item in flattened]
        else:
            attributes["value"] = [attributes["weight"] + ("" if item.startswith("-") else "+") + item for item in flattened]

    memo[attributes["graph_node"]+"_"+str(attributes["iteration"])+str(attributes["start"])] = attributes["value"]

    return

def run_algorithm(dot_file, queuesize = 1):
    G = read_dot_file(dot_file)

    source = [node for node in G.nodes if G.in_degree(node) == 0][0]
    sink = [node for node in G.nodes if G.out_degree(node) == 0][0]

    node_connections = {
    node: {
        "predecessors": list(G.predecessors(node)),
        "successors": list(G.successors(node)),
    }
    for node in G.nodes
    }

    tree = nx.DiGraph()

    tree.add_node("root", graph_node=sink, iteration=0, value="", weight=sink+"_"+str(0), start=False)  # Add root node with attributes

    memo  = {}

    paths = nx.all_simple_paths(G, source, sink)

    paths, paths2 = itertools.tee(paths, 2)

    depth_dict = defaultdict(set)
    
    for path in paths2:
        for depth, node in enumerate(path):
            depth_dict[depth].add(node)

    checker = SuccessorChecker(G, depth_dict, queuesize)

    dfs_post_order(tree, queuesize, "root", node_connections, 0, memo, depth_dict, checker, sink)

    return tree.nodes["root"]["value"]

# After main algorithm returns, we maximize the resulting expressions
def compute_max(expressions, constraints):
    overall_max = 0
    for expr in expressions:
        sum = 0
        terms = expr.split("+")
        for i in range(len(terms)-1):
            key = terms[i].split("_", 1)[0]
            sum += constraints[key]

        if terms[-1].endswith("!"):
            sum -= constraints[terms[-1].split("_", 1)[0]]

        if sum > overall_max:
            overall_max = sum

    return overall_max

def extract_wcet(dot_file):
    wcet_dict = {}
    with open(dot_file, "r") as f:
        for line in f:
            match = re.search(r'(\w+)\s*\[WCET=(\d+)\]', line)
            if match:
                wcet_dict[match.group(1)] = int(match.group(2))
    return wcet_dict


def main(dot_file, queuesize = 1, timing = True):

    # Measure runtime of both main algorithm and maximization steps
    if timing:
        start1 = time.perf_counter()

    expressions = run_algorithm(dot_file, queuesize)

    constraints = extract_wcet(dot_file)

    if timing:
        end1 = time.perf_counter()

        start2 = time.perf_counter()


    ret = compute_max(expressions, constraints)

    if timing:
        end2 = time.perf_counter()

        return (end1 - start1), (end2 - start2), len(expressions)

    return ret