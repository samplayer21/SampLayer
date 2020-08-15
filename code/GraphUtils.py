import gc
from operator import itemgetter

import pandas as pd
from igraph import Graph

graph = None


class GraphUtils():
    @staticmethod
    def load_graph(file_path, sep=" ", directed=True):
        gc.enable()
        # print("Reading CSV file")
        df = pd.read_csv(file_path, sep=sep, header=None, names=["source", "target"], comment="#")
        vertices = pd.unique(df[["source", "target"]].values.ravel('K'))
        v2idx = {v: i for i, v in enumerate(vertices)}
        df["source"] = df["source"].apply(lambda r: v2idx[r])
        df["target"] = df["target"].apply(lambda r: v2idx[r])
        vertices = list(range(len(vertices)))
        g = Graph(directed=directed)
        # print("Adding Vertices")
        g.add_vertices(vertices)
        # print("Adding edges")
        g.add_edges(df.values)
        # print(f"Done loading the graph. Number of nodes: {g.vcount()}, number of edges: {g.ecount()}")
        del df

        # sort degrees in descending order of degrees, and then by ascending vertex names
        in_degrees = [(v, -deg) for v, deg in zip(vertices, g.indegree(vertices))]
        in_degrees = sorted(in_degrees, key=lambda x: (x[1], x[0]))
        in_degrees = [(v, -deg) for v, deg in in_degrees]
        return g, in_degrees

    @staticmethod
    def get_high_nodes(g, in_degrees=None, min_in_deg=None, num_high_nodes=None):
        assert (min_in_deg is None or num_high_nodes is None)
        if in_degrees is None:
            in_degrees = sorted(enumerate(g.indegree()), key=itemgetter(1), reverse=True)
        high_nodes = [i for i, in_deg in in_degrees if in_deg >= min_in_deg] if min_in_deg else [i for i, _ in
                                                                                                 in_degrees[
                                                                                                 :num_high_nodes]]
        return high_nodes

    @staticmethod
    def get_high_graph(g, in_degrees=None, min_in_deg=None, num_high_nodes=None):
        assert (min_in_deg is None or num_high_nodes is None)
        if in_degrees is None:
            in_degrees = sorted(enumerate(g.indegree()), key=itemgetter(1), reverse=True)
        high_nodes = [i for i, in_deg in in_degrees if in_deg >= min_in_deg] if min_in_deg else [i for i, _ in
                                                                                                 in_degrees[
                                                                                                 :num_high_nodes]]
        return g.subgraph(high_nodes)

    @staticmethod
    def get_largest_cc_in_high_graph(g, in_degrees, num_high_nodes):
        high_graph = GraphUtils.get_high_graph(g, in_degrees, num_high_nodes=num_high_nodes)
        return [high_graph.vs[i]["name"] for i in max(high_graph.clusters(mode="WEAK"), key=len)]
