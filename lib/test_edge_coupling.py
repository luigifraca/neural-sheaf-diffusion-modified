import pytest
import torch

from lib.edge_coupling import (
    laplacian_matrix_to_edge_weights,
    sort_edge_index_with_values,
    sort_sparse_entries,
    validate_edge_index,
)


def test_sort_sparse_entries_carries_values_with_indices():
    indices = torch.tensor([[2, 0, 1], [0, 2, 0]])
    values = torch.tensor([20.0, 2.0, 10.0])

    sorted_indices, sorted_values = sort_sparse_entries(indices, values, width=3)

    assert sorted_indices.tolist() == [[0, 1, 2], [2, 0, 0]]
    assert sorted_values.tolist() == [2.0, 10.0, 20.0]


def test_sort_edge_index_with_values_carries_map_rows():
    edge_index = torch.tensor([[2, 0, 1], [0, 2, 0]])
    values = torch.tensor([[20.0], [2.0], [10.0]])

    sorted_edge_index, sorted_values = sort_edge_index_with_values(edge_index, values)

    assert sorted_edge_index.tolist() == [[0, 1, 2], [2, 0, 0]]
    assert sorted_values.squeeze(-1).tolist() == [2.0, 10.0, 20.0]


def test_laplacian_edge_weights_are_keyed_by_edge_not_sparse_order():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    laplacian = torch.tensor([
        [1, 0, 2, 1, 0, 1, 2],
        [2, 0, 1, 1, 1, 0, 2],
        [-3.0, 5.0, -3.0, 7.0, -2.0, -2.0, 8.0],
    ])

    edge_weights, entries = laplacian_matrix_to_edge_weights(
        laplacian,
        num_nodes=3,
        stalk_dim=1,
        edge_index=edge_index,
        return_entries=True,
    )

    assert list(edge_weights) == [(0, 1), (1, 2)]
    assert edge_weights[(0, 1)] == pytest.approx(2.0)
    assert edge_weights[(1, 2)] == pytest.approx(3.0)
    assert entries[(0, 1)] == [(0, 1, 2.0), (1, 0, 2.0)]
    assert entries[(1, 2)] == [(1, 2, 3.0), (2, 1, 3.0)]


def test_laplacian_edge_weights_reject_edge_index_mismatch():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    laplacian = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
        [-2.0, -2.0, -3.0, -3.0],
    ])

    with pytest.raises(ValueError, match="does not match edge_index"):
        laplacian_matrix_to_edge_weights(laplacian, num_nodes=3, stalk_dim=1, edge_index=edge_index)


def test_validate_edge_index_rejects_missing_reverse_edges():
    with pytest.raises(ValueError, match="missing reverse"):
        validate_edge_index(torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
