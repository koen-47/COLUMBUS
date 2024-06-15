from util import get_answer_graph_pairs

graphs, compound_graphs = get_answer_graph_pairs()
graphs.update(compound_graphs)

graphs_no_icons, graphs_icons = {}, {}
for answer, graph in graphs.items():
    pass
