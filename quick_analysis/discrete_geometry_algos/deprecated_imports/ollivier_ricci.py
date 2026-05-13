__author__ = "Patrick R. Nicolas"
__copyright__ = "Copyright 2023, 2026  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
from typing import List, Tuple, Optional, Callable
# 3rd Party imports
import torch
# Library imports
from .floyd_warshall import FloydWarshall
from .sinkhorn_knopp import SinkhornKnopp

__all__ = ['OllivierRicci']


def _debug_tensor(name: str, tensor: torch.Tensor, debug: bool) -> None:
    if debug:
        print(f'[DEBUG] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}')
        print(tensor)


class OllivierRicci(FloydWarshall):
    """
    Implementation of the computation of the Ollivier-Ricci Curvature

    Over-squashing and over-smoothing represent the primary obstacles in training Graph Neural Networks (GNNs). While
    over-squashing generates critical information bottlenecks, over-smoothing leads to the excessive generalization of
    node features. By utilizing Ollivier-Ricci curvature, researchers can identify and remediate graph regions that
    obstruct effective information flow.
    Applied to graphs, Ollivier-Ricci Curvature (ORC) serves as a discrete approximation that uncovers local topology
    and geometric properties through the lens of optimal transport.

    Let G be a graph with a shortest-path metric (a.k.a. cost matrix) d, mu the probability measure on graph for a
    given node v. The Ollivier-Ricci curvature of a pair of node (i, j) is computed from the Wasserstein distance W
    .. math::
        \kappa_{OR}(i, j)= 1-\frac{W_{1}(\mu _{i}, \mu _{j})}{d(i, j)} \    \    \     \    [1]


    Reference Article: Curvature-informed Graph Learning https://patricknicolas.substack.com/publish/post/181931881

    This class has two constructors:
    __init__:  Default for which the user provides optionally the weights of the edges of the graph
    build: Alternative constructor that generate the edge weights from the closed form geodesic distance between two
        nodes laying into the underlying manifold.
    """
    __slots__ = [
        'adjacency',
        'wasserstein_1_approximation',
        'alpha',
        'limit_ricci',
        'true_weights_distribution',
        'true_weights_distance',
        'nodes_num',
        'debug'
    ]

    def __init__(self,
                 edge_index: List[Tuple[int, int]],
                 weights: Optional[torch.Tensor],
                 epsilon: float,
                 rc: Tuple[torch.Tensor, torch.Tensor] = None,
                 alpha: float = 0.4,
                 limit_ricci: bool = False,
                 true_weights_distribution: bool = False,
                 true_weights_distance: bool = False,
                 nodes_num: Optional[int] = None,
                 debug: bool = False) -> None:
        """
        Constructor for the Olliver-Ricci curvature. It is assumed that the graph is undirected.

        @param edge_index: List of pairs (tuples) (index source node, index destination node)
        @type edge_index: Tuple[int, int]
        @param weights: Optional weights associated with the weights
        @type weights: torch.Tensor
        @param epsilon: Entropy regularization scale factor
        @type epsilon: float
        @param alpha: Probability mass assigned to the center node in the lazy neighborhood measure
        @type alpha: float
        @param limit_ricci: If True, return curvature / (1 - alpha) to approximate the alpha -> 1 limit
        @type limit_ricci: bool
        @param true_weights_distribution: If True, compute lazy random-walk measures from edge weights instead of binary degrees
        @type true_weights_distribution: bool
        @param true_weights_distance: If True, use actual weights for distance computation instead of normalized ones
        @type true_weights_distance: bool
        @param nodes_num: Optional total number of graph nodes, useful when isolated nodes should be represented
        @type nodes_num: int
        @param debug: If True, print intermediate tensors and shapes
        @type debug: bool
        @param rc: Pair of marginal distributions for rows (r) and columns (c) of the joint distribution matrix used
                    for the Wasserstein distance
        @type rc: Tuple[Tensor, Tensor]
        """
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f'Alpha {alpha} should be [0.0, 1.0]')
        if limit_ricci and alpha == 1.0:
            raise ValueError('Alpha should be < 1.0 when limit_ricci is True')
        if true_weights_distribution and weights is None:
            raise ValueError('To have a true-weights probability distribution you should provide a weights Tensor')
        if true_weights_distance and weights is None:
            raise ValueError('To have a true-weights distance you should provide a weights Tensor')

        super(OllivierRicci, self).__init__(
            edge_index=edge_index,
            is_undirected=True,
            weights=weights,
            true_weights_distance=true_weights_distance,
            debug=debug
        )

        self.nodes_num = nodes_num
        self.true_weights_distribution = true_weights_distribution
        self.true_weights_distance = true_weights_distance
        self.limit_ricci = limit_ricci
        self.alpha = alpha
        self.debug = debug
        self.adjacency = FloydWarshall.create_adjacency(edge_index=edge_index, is_indirect=True)
        _debug_tensor('OllivierRicci.adjacency', self.adjacency, self.debug)
        (r, c) = rc if rc is not None else self.__get_marginal_distributions(self.adjacency, alpha)
        _debug_tensor('OllivierRicci.r.selected', r, self.debug)
        _debug_tensor('OllivierRicci.c.selected', c, self.debug)
        self.wasserstein_1_approximation = SinkhornKnopp.build(r, c, self, epsilon, self.debug)

    @classmethod
    def build(cls,
              edge_index: List[Tuple[int, int]],
              geodesic_distance: Callable[[int], torch.Tensor],
              epsilon: float,
              rc: Tuple[torch.Tensor, torch.Tensor] = None,
              alpha: float = 0.4,
              limit_ricci: bool = False,
              true_weights_distribution: bool = False,
              true_weights_distance: bool = False,
              nodes_num: Optional[int] = None,
              debug: bool = False):
        """
        Alternative constructor for the computation of the Olliver-Ricci curvature of a mesh or a graph. Contrary
        to the default constructor, this method take a closed-form of the geodesic distance on the underlying
        manifold and generate the weights for each edge.

        @param edge_index:  List of pairs (tuples) (index source node, index destination node)
        @type edge_index: Tuple[int, int]
        @param geodesic_distance: Closed formula for the geodesic distance of the underlying manifold
        @type geodesic_distance: Callable[[int], torch.Tensor]
        @param epsilon: Entropy regularization scale factor
        @type epsilon: float
        @param alpha: Probability mass assigned to the center node in the lazy neighborhood measure
        @type alpha: float
        @param limit_ricci: If True, return curvature / (1 - alpha) to approximate the alpha -> 1 limit
        @type limit_ricci: bool
        @param true_weights_distribution: If True, compute lazy random-walk measures from edge weights instead of binary degrees
        @type true_weights_distribution: bool
        @param true_weights_distance: If True, uses actual weights as distance factors instead of normalised ones
        @type true_weights_distance: bool
        @param nodes_num: Optional total number of graph nodes, useful when isolated nodes should be represented
        @type nodes_num: int
        @param debug: If True, print intermediate tensors and shapes
        @type debug: bool
        @param rc: Pair of marginal distributions for rows (r) and columns (c) of the joint distribution matrix used
                    for the Wasserstein distance
        @type rc: Tuple[Tensor, Tensor]
        @return: Instance of this class
        @rtype: OllivierRicci
        """
        weights = geodesic_distance(len(edge_index))
        return cls(
            edge_index,
            weights,
            epsilon,
            rc,
            alpha,
            limit_ricci,
            true_weights_distribution,
            true_weights_distance,
            nodes_num,
            debug
        )

    def curvature(self, n_iters: int, early_stop_threshold: float) -> torch.Tensor:
        """
        Method that compute the curvature of a graph or mesh using the Olliver-Ricci formula:
            K = 1 - W/d
        W; Approximate 1-dimensional Wasserstein distance using the iterative Sinkhorn-Knopp algorithm
        d: Distance of the shortest path between any given nodes using the Floyd_Warshall formula

        @param n_iters: Maximum number of iterations allowed
        @type n_iters: int
        @param early_stop_threshold: Early stopping condition
        @type early_stop_threshold: float
        @return: Discrete curvature
        @rtype: torch.Tensor
        """
        # Load the shortest paths as the cost matrix in the Wasserstein distance
        shortest_paths = self.wasserstein_1_approximation.cost_matrix
        _debug_tensor('OllivierRicci.shortest_paths_for_curvature', shortest_paths, self.debug)
        curvature = torch.zeros_like(shortest_paths)

        edges = torch.nonzero(self.adjacency)
        _debug_tensor('OllivierRicci.edges', edges, self.debug)
        for u, v in edges:
            # Compute the approximate Wasserstein distance - Numerator
            num_iters, w1 = self.wasserstein_1_approximation(n_iters, early_stop_threshold)
            # Load the all-pairs shortest path between u and v nodes
            shortest_path_uv = shortest_paths[u, v]
            if self.debug:
                print(
                    f'[DEBUG] OllivierRicci.edge=({u.item()}, {v.item()}), '
                    f'num_iters={num_iters}, w1={w1}, shortest_path_uv={shortest_path_uv}'
                )
            # Apply the Olliver-Ricci formula
            curvature[u, v] = 1 - (w1 / shortest_path_uv)
            if self.is_undirected:
                curvature[v, u] = curvature[u, v]
        if self.limit_ricci:
            curvature = curvature / (1 - self.alpha)
        _debug_tensor('OllivierRicci.curvature', curvature, self.debug)
        return curvature

    """ -------------------------  Private Helper Methods -------------------------  """

    def __get_marginal_distributions(self, adjacency: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        joint_probability_measures = self.__compute_prob_measures_true_weights(alpha) \
            if self.true_weights_distribution else OllivierRicci.__compute_prob_measures(adjacency, alpha)
        _debug_tensor('OllivierRicci.joint_probability_measures', joint_probability_measures, self.debug)
        # Extract the marginal distribution from the joint distribution
        r = joint_probability_measures.sum(dim=1)
        c = joint_probability_measures.sum(dim=0)
        _debug_tensor('OllivierRicci.r.from_joint_rowsum', r, self.debug)
        _debug_tensor('OllivierRicci.c.from_joint_colsum', c, self.debug)
        return r, c

    @staticmethod
    def __compute_prob_measures(adjacency: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
        n = adjacency.shape[0]
        # Define neighborhood distributions m_x
        # m_x(v) = alpha if v=x, else (1-alpha)/degree if v is neighbor
        degrees = adjacency.sum(dim=1)
        eye = torch.eye(n)

        # Probability measures for all nodes: (N, N)
        probs = (alpha * eye) + ((1 - alpha) * adjacency / degrees.clamp_min(1).unsqueeze(1))
        isolated_nodes = degrees == 0
        probs[isolated_nodes] = eye[isolated_nodes]
        return probs

    def __compute_prob_measures_true_weights(self, alpha: float = 0.4) -> torch.Tensor:
        n_nodes = self.nodes_num if self.nodes_num is not None else max(sum(self.edge_index, ())) + 1
        A = torch.zeros((n_nodes, n_nodes), dtype=self.weights.dtype, device=self.weights.device)
        edge_index = torch.tensor(self.edge_index, device=self.weights.device)

        i = edge_index[:, 0]
        j = edge_index[:, 1]

        A[i, j] = self.weights
        A[j, i] = self.weights
        _debug_tensor('OllivierRicci.weighted_adjacency', A, self.debug)
        weighted_degrees = A.sum(dim=1)
        _debug_tensor('OllivierRicci.weighted_degrees', weighted_degrees, self.debug)
        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

        probs = (alpha * eye) + ((1 - alpha) * A / weighted_degrees.clamp_min(1e-12).unsqueeze(1))
        isolated_nodes = weighted_degrees == 0
        probs[isolated_nodes] = eye[isolated_nodes]
        return probs
