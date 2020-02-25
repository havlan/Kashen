from graphviz import Digraph, Source

base_filename = "C:/Users/havar/Home/cache_simulation_results/"


def _visualize_d_tree(filename):
    dot = Source.from_file(filename)
    dot.view()


if __name__ == '__main__':
    _visualize_d_tree(base_filename + "d3.gv")
