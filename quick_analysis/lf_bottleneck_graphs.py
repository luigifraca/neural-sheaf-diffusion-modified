from dataclasses import dataclass
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


@dataclass
class BottleneckGraphConfig:
    n_left: int = 80
    n_right: int = 80
    bridge_width: int = 1
    feature_dim: int = 16
    num_classes: int = 2
    homophily: float = 0.8
    seed: int = 0


class BottleneckGraphGenerator:
    def __init__(self, config: BottleneckGraphConfig):
        self.cfg = config
        self.rng = torch.Generator().manual_seed(config.seed)

    def build_topology(self) -> nx.Graph:
        c = self.cfg

        G_left = nx.complete_graph(c.n_left)
        G_right = nx.complete_graph(c.n_right)
        G_right = nx.relabel_nodes(G_right, lambda i: i + c.n_left)

        G = nx.compose(G_left, G_right)

        # Mark all existing clique edges as non-bottleneck
        nx.set_edge_attributes(G, False, "bottleneck")

        left_nodes = list(range(c.n_left))
        right_nodes = list(range(c.n_left, c.n_left + c.n_right))

        for k in range(c.bridge_width):
            u = left_nodes[k % len(left_nodes)]
            v = right_nodes[k % len(right_nodes)]
            G.add_edge(u, v, bottleneck=True)

        return G

    def build_labels(self, num_nodes: int) -> torch.Tensor:
        c = self.cfg

        y = torch.zeros(num_nodes, dtype=torch.long)
        y[c.n_left:] = 1

        if c.num_classes > 2:
            y = torch.randint(
                low=0,
                high=c.num_classes,
                size=(num_nodes,),
                generator=self.rng
            )

        return y

    def build_features(self, y: torch.Tensor) -> torch.Tensor:
        c = self.cfg

        class_centres = torch.randn(
            c.num_classes,
            c.feature_dim,
            generator=self.rng
        )

        x = class_centres[y] + 0.25 * torch.randn(
            y.size(0),
            c.feature_dim,
            generator=self.rng
        )

        return x.float()

    def generate(self) -> Data:
        G = self.build_topology()

        # Convert NetworkX graph to PyG Data
        data = from_networkx(G)

        y = self.build_labels(data.num_nodes)
        x = self.build_features(y)

        data.x = x
        data.y = y

        # Useful metadata
        data.num_classes = self.cfg.num_classes
        data.bridge_width = self.cfg.bridge_width

        return data