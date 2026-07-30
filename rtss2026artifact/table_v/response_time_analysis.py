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
        node = node.strip('"')
        G.add_node(node, **data)
    
    for u, v, data in graph.edges(data=True):
        u, v = u.strip('"'), v.strip('"') 
        G.add_edge(u, v, **data)

    return G

import itertools
from collections import defaultdict
import networkx as nx
import re


import itertools
import re
import networkx as nx
from collections import defaultdict

def logical_depth_dict(G, source):
    reachability_graph = nx.DiGraph()
    reachability_graph.add_nodes_from(G.nodes())

    def add_edge(u, v, weight):
        if reachability_graph.has_edge(u, v):
            old_weight = reachability_graph[u][v]["weight"]
            reachability_graph[u][v]["weight"] = min(old_weight, weight)
        else:
            reachability_graph.add_edge(u, v, weight=weight)

    # Data-dependency edges stay within the same iteration.
    for u, v in G.edges():
        add_edge(u, v, 0)

    for node in G.nodes():
        node_type = G.nodes[node].get("type", "")

        # Sequential-execution edges move forward by one iteration.
        if node_type == "syncpost":
            syncpre = re.sub(r"(-?prime)$", "", node)
            add_edge(node, syncpre, 1)
        elif node_type in ["async", "GPU"]:
            add_edge(node, node, 1)

        # Downstream-blocking edges move opposite to CPU-side data flow and
        # forward by one iteration.
        if node_type == "async":
            for successor in G.successors(node):
                if "GPU" not in successor:
                    add_edge(successor, node, 1)
        elif node_type == "syncpre":
            for successor in G.successors(node + "-prime"):
                add_edge(successor, node, 1)

    node_depth = nx.single_source_dijkstra_path_length(
        reachability_graph.reverse(copy=False), source, weight="weight"
    )

    depth_dict = defaultdict(set)
    for node, depth in node_depth.items():
        depth_dict[depth].add(node)

    return depth_dict

# For the path specific value
class SuccessorChecker:
    def __init__(self, graph, level_map):
        # Map each node to its minimum level
        self.node_to_level = {}
        for level, nodes in level_map.items():
            for node in nodes:
                if node not in self.node_to_level:
                    self.node_to_level[node] = level
                else:
                    self.node_to_level[node] = min(self.node_to_level[node], level)

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


def dfs_post_order(tree, node, node_properties, iteration, memo, checker):

    attributes = tree.nodes[node]

    # Check if self has been memoized
    if attributes["graph_node"]+"_"+str(attributes["iteration"])+str(attributes["start"]) in memo:
        attributes["value"] = memo[attributes["graph_node"]+"_"+str(attributes["iteration"])+str(attributes["start"])]
        return

    # Create next level of tree if we are a starting node, else add start
    if attributes["start"]:
        # Autoconcurrency edges
        if node_properties[attributes["graph_node"]]["type"] == "syncpre":
            id = uuid.uuid4()
            tree.add_node(id, graph_node=attributes["graph_node"]+"-prime", iteration=iteration+1, value="", weight=attributes["graph_node"]+"-prime"+"_"+str(iteration+1), start=False)  
            tree.add_edge(node, id)
        elif node_properties[attributes["graph_node"]]["type"] != "syncpost":
            id = uuid.uuid4()
            tree.add_node(id, graph_node=attributes["graph_node"], iteration=iteration+1, value="", weight=attributes["graph_node"]+"_"+str(iteration+1), start=False)  
            tree.add_edge(node, id)

        # Normal edges
        for pred in node_properties[attributes["graph_node"]]["predecessors"]:
            id = uuid.uuid4()
            tree.add_node(id, graph_node=pred, iteration=iteration, value="", weight=pred+"_"+str(iteration), start=False)  
            tree.add_edge(node, id)

        # Downstream edges
        # GPU nodes and syncpost nodes never have incoming downstream edges
        if node_properties[attributes["graph_node"]]["type"] == "async":
            for succ in node_properties[attributes["graph_node"]]["successors"]:
                if "GPU" not in succ:
                    id = uuid.uuid4()
                    tree.add_node(id, graph_node=succ, iteration=iteration+1, value="", weight="", start=True)  
                    tree.add_edge(node, id)
        elif node_properties[attributes["graph_node"]]["type"] == "syncpre":
            for succ in node_properties[attributes["graph_node"]+"-prime"]["successors"]:
                id = uuid.uuid4()
                tree.add_node(id, graph_node=succ, iteration=iteration+1, value="", weight="", start=True)  
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
        if childattributes["start"] == False:
            if node_properties[childattributes["graph_node"]]["type"] in ["syncpre", "async"]:
                # I have a 0 cost path to source (downstream)
                if checker.is_at_level(childattributes["graph_node"], childattributes["iteration"]) :
                    #But I also have a path to someone else with a 0 cost path to source
                    if checker.has_successor_in_lower_or_equal_level(childattributes["graph_node"], childattributes["iteration"]):
                        tree.nodes[child]["value"] = childname+"!"       
                    else:
                        tree.nodes[child]["value"] = childname                

                # I have a path to someone with a 0 cost path to source (data dependency)
                elif checker.has_successor_in_lower_or_equal_level(childattributes["graph_node"], childattributes["iteration"]):
                    tree.nodes[child]["value"] = childname+"!"

                # I have a path to another version of me which has a 0 cost path to source (autoconcurrency)
                elif checker.is_strictly_lower_level(childattributes["graph_node"], childattributes["iteration"]):
                    tree.nodes[child]["value"] = childname+"!"

            elif node_properties[childattributes["graph_node"]]["type"] == "syncpost":
                # I have a path to someone with a 0 cost path to source (data dependency)
                if checker.has_successor_in_lower_or_equal_level(childattributes["graph_node"], childattributes["iteration"]):
                    tree.nodes[child]["value"] = childname+"!"

                # I have a path to my syncpre which has a 0 cost path to source (autoconcurrency)
                if checker.is_at_level(childattributes["graph_node"].strip("-prime"), childattributes["iteration"]-1):
                        tree.nodes[child]["value"] = childname+"!"       

                # I have a path to my syncpre which has a path to another version of itself with a 0 cost path to source (autoconcurrency)
                elif checker.is_strictly_lower_level(childattributes["graph_node"].strip("-prime"), childattributes["iteration"] - 1):
                    tree.nodes[child]["value"] = childname + "!"

            # GPU node
            else:
                # I have a path to someone with a 0 cost path to source (data dependency)
                if checker.has_successor_in_lower_or_equal_level(childattributes["graph_node"], childattributes["iteration"]):
                    tree.nodes[child]["value"] = childname+"!"

                # I have a path to my syncpre (via my syncpost) which has a 0 cost path to source (autoconcurrency)
                if node_properties[childattributes["graph_node"].removeprefix("GPU-")]["type"] == "syncpre":
                    if checker.is_at_level(childattributes["graph_node"].removeprefix("GPU-"), childattributes["iteration"]-1):
                        tree.nodes[child]["value"] = childname+"!"       


            if tree.nodes[child]["value"] == "":
                dfs_post_order(tree, child, node_properties, childattributes["iteration"], memo, checker)
            else:
                pass


        else:
            dfs_post_order(tree, child, node_properties, childattributes["iteration"], memo, checker)


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

def run_algorithm(dot_file):
    G = read_dot_file(dot_file)

    source = [node for node in G.nodes if G.in_degree(node) == 0][0]
    sink = [node for node in G.nodes if G.out_degree(node) == 0][0]

    node_properties = {
    node: {
        "predecessors": list(G.predecessors(node)),
        "successors": list(G.successors(node)),
        "type": G.nodes[node].get("type", "").strip('"').strip()
    }
    for node in G.nodes
    }

    tree = nx.DiGraph()

    tree.add_node("root", graph_node=sink, iteration=0, value="", weight=sink+"_"+str(0), start=False)  # Add root node with attributes

    memo  = {}

    checker = SuccessorChecker(G, logical_depth_dict(G, source))

    dfs_post_order(tree, "root", node_properties, 0, memo, checker)

    return tree.nodes["root"]["value"]

# After main algorithm returns, we maximize the resulting expressions
def compute_max(expressions, constraints):
    overall_max = float('-inf')

    for expr in expressions:
        total = 0
        terms = expr.split("+")
        for i in range(len(terms) - 1):
            key = terms[i].split("_", 1)[0]
            total += constraints[key]

        # For the path specific value
        if terms[-1].endswith("!"):
            pass
        else:
            total += constraints[terms[-1].split("_", 1)[0]]

        if total > overall_max:
            overall_max = total

    return overall_max


def extract_wcet(dot_file):
    wcet_dict = {}
    with open(dot_file, "r") as f:
        for line in f:
            match = re.search(r'"?([\w\'-]+)"?\s*\[.*?WCET\s*=\s*"?(\d+)"?', line)
            if match:
                node = match.group(1).strip('"').strip()
                value = int(match.group(2))
                wcet_dict[node] = value
    return wcet_dict


def main(dot_file, timing = False):
    # Measure runtime of both main algorithm and maximization steps
    if timing:
        start1 = time.perf_counter()

    expressions = run_algorithm(dot_file)

    constraints = extract_wcet(dot_file)

    if timing:
        end1 = time.perf_counter()

        start2 = time.perf_counter()

    ret = compute_max(expressions, constraints)

    if timing:
        end2 = time.perf_counter()

        return (end1 - start1), (end2 - start2)

    return ret
