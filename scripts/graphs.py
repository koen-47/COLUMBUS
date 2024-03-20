import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a MultiDiGraph
G = nx.MultiDiGraph()

# Add nodes
G.add_nodes_from([1, 2])

# Add multiple edges between the same nodes with different attributes
G.add_edge(1, 2, key=1, color='red', label='First Edge')
G.add_edge(1, 2, key=2, color='blue', label='Second Edge')

# Set up the plot
plt.figure(figsize=(8, 6))

# Draw nodes
pos = nx.spring_layout(G, seed=42)  # You can choose a layout algorithm of your choice
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

# Draw edges with labels and different attributes
for edge in G.edges(keys=True):
    edge_attrs = G.edges[edge[0], edge[1], edge[2]]
    nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2, edge_color=edge_attrs['color'], connectionstyle=f"arc3,rad=0.2")

# Draw edge labels manually for each edge
for edge in G.edges(keys=True):
    edge_attrs = G.edges[edge[0], edge[1], edge[2]]
    label_pos = (pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2
    dx = pos[edge[1]][0] - pos[edge[0]][0]
    dy = pos[edge[1]][1] - pos[edge[0]][1]
    angle = np.arctan2(dy, dx)
    if angle > np.pi / 2 or angle < -np.pi / 2:
        angle += np.pi
    plt.text(label_pos[0], label_pos[1], edge_attrs['label'], horizontalalignment='center', verticalalignment='center', color='black',
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'), rotation=np.degrees(angle))

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Adjust spacing to avoid overlap
plt.tight_layout()

# Show the plot
plt.axis('off')
plt.show()
