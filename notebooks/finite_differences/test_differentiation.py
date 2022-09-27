"""
Tests for finite difference functions and classes
"""
import numpy as np
import numpy.testing as npt
import pytest

from .differentiation import build_diff_from_node_to_cell


def test_diff_from_node_to_cell_matrix():
    """
    Test build_diff_from_node_to_cell with a small matrix
    """
    n_nodes = 4
    node_spacing = 0.25
    diff_matrix = build_diff_from_node_to_cell(node_spacing, n_nodes)
    expected_matrix = np.array(
        [
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
        ],
        dtype=float,
    )
    expected_matrix /= node_spacing
    npt.assert_allclose(diff_matrix.toarray(), expected_matrix)


@pytest.mark.parametrize("n_nodes", 2 ** (3 + np.arange(6)))
def test_differentiation_using_diff_from_node_to_cell_matrix(n_nodes):
    """
    Test finite difference differentiation using build_diff_from_node_to_cell
    """
    # Define a regular mesh between 0 and 1
    nodes = np.linspace(0, 1, n_nodes)
    # Sample a sinusoidal function on the nodes of the mesh
    sine = np.sin(2 * np.pi * nodes)
    # Build the differentiation matrix
    node_spacing = nodes[1] - nodes[0]  # get node spacing
    diff_matrix = build_diff_from_node_to_cell(node_spacing, n_nodes)
    # Compute the derivative of the sine on the cell centers
    diff_sine = diff_matrix @ sine
    # Build the expected values
    cell_centers = (nodes[1:] + nodes[:-1]) / 2
    expected = 2 * np.pi * np.cos(2 * np.pi * cell_centers)
    # Get the average error
    average_error = np.abs(diff_sine - expected).mean()
    # Check if the average error is lower than the node_spacing (the expected
    # error of the numerical differentiation)
    assert average_error < node_spacing
