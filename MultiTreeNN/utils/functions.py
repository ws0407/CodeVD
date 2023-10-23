from collections import OrderedDict
import subprocess
from .objects import Function

# -- digraph -- #
def create_digraph(name, nodes):
    with open('digraph.gv', 'w') as f:
        f.write("digraph {\n")
        digraph = ''

        for n_id, node in nodes.items():
            label = node.get_code() + "\n" + node.label
            f.write(f"\"{node.get_code()}\" [label=\"{label}\"]\n")
            for e_id, edge in node.edges.items():
                if edge.type != "Ast": continue

                if edge.node_in in nodes and edge.node_in != node.id:
                    n_in = nodes.get(edge.node_in)
                    label = "\"" + n_in.label + "\""
                    digraph += f"\"{node.get_code()}\" -> \"{n_in.get_code()}\" [label={label}]\n"
                '''
				if edge.node_out in nodes and edge.node_out != node.id:
					n_out = nodes.get(edge.node_out)
					label = "\"" + n_out.label + "\""
					digraph += f"\"{n_out.get_code()}\" -> \"{node.get_code()}\" [label={label}]\n"		
				'''
        f.write(digraph)
        f.write("}\n")
    subprocess.run(["dot", "-Tps", "digraph.gv", "-o", f"{name}.ps"], shell=False)


'''
def to_digraph(name, nodes):
    k_nodes = nodes.keys()
    code = {}
    connections = { "in" : dict.fromkeys(k_nodes), "out" : dict.fromkeys(k_nodes) }

	for n_id, node in nodes.items():
		#print(n_id, node.properties)
		connections = node.connections(connections, "Ast")
		code.update({n_id : node.get_code()})

	create_digraph(name, code, k_nodes, connections)
'''

# -- cpg -- #
def order_nodes(nodes, max_nodes):
    # sorts nodes by line and column

    # nodes_by_column = sorted(nodes.items(), key=lambda n: n[1].get_column_number())
    nodes_by_line = sorted(nodes.items(), key=lambda n: n[1].get_line_number())

    for i, node in enumerate(nodes_by_line):
        node[1].order = i

    if len(nodes) > max_nodes:
        print(f"CPG cut - original nodes: {len(nodes)} to max: {max_nodes}")
        return OrderedDict(nodes_by_line[:max_nodes])

    return OrderedDict(nodes_by_line)


def filter_nodes(nodes):
    return {n_id: node for n_id, node in nodes.items() if node.has_code() and
            node.has_line_number() and
            node.label not in ["Comment", "Unknown"]}


def parse_to_nodes(cpg, max_nodes=500):
    nodes = {}
    for function in cpg["functions"]:
        func = Function(function)
        # Only nodes with code and line number are selected
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)

    return order_nodes(nodes, max_nodes)
