import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    # print(pos)
    # input()
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  
    if len(children) != 0:
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap,
                                 xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

def treeplot(adj_matrix, ax=None):
    # adj_matrix = np.array(adj_matrix_list)

    # Convert adjacency matrix to NetworkX graph
    # G = nx.from_numpy_array(adj_matrix)
    # G = nx.from_scipy_sparse_matrix(adj_matrix)
    # G = nx.to_scipy_sparse_array(adj_matrix)

    # Convert the sparse matrix to a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(adj_matrix.shape[0]))
    rows, cols = adj_matrix.nonzero()
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)

    # Assuming node 0 is the root
    root = 0

    # Get hierarchical layout
    pos = hierarchy_pos(G, root)

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_size=2, node_color='black', font_size=10,
            font_weight='bold', edge_color='black', ax=ax)
    # plt.show(block=True)

