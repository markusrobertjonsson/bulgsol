import networkx as nx


def binary_tree_layout(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, 
                  pos = None, parent = None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.
       each node has an attribute "left: or "right"'''
    if pos == None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root))
    if parent != None:
        neighbors.remove(parent)
    if len(neighbors)!=0:
        dx = width/2.
        leftx = xcenter - dx/2
        rightx = xcenter + dx/2
        for neighbor in neighbors:
            if G.nodes[neighbor]['child_status'] == 'left':
                pos = binary_tree_layout(G,neighbor, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=leftx, pos=pos, 
                    parent = root)
            elif G.nodes[neighbor]['child_status'] == 'right':
                pos = binary_tree_layout(G,neighbor, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=rightx, pos=pos, 
                    parent = root)
    return pos


G= nx.Graph()
G.add_edges_from([(0,1),(0,2), (1,3), (1,4), (2,5), (2,6), (3,7)])
for node in G.nodes():
    if node%2==0:
        G.nodes[node]['child_status'] = 'left'  #assign even to be left
    else:
        G.nodes[node]['child_status'] = 'right' #and odd to be right
pos = binary_tree_layout(G,0)
nx.draw(G, pos=pos, with_labels = True)
