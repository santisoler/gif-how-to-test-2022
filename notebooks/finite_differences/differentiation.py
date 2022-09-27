"""
Functions that build differentiation matrices
"""
import numpy as np
from scipy import sparse


def build_diff_from_node_to_cell(node_spacing, n_nodes):
    r"""
    Build matrix for 1D differentiation from nodes to cell centers

    This function builds a matrix :math:`\mathbf{D}` that represents the
    finite differentiation operator. It operates over a discretized function on
    the mesh nodes and returns the derivative on the mesh cell centers.

    .. math::

        D u \Bigg_{i + \frac{1}{2}} =
            \frac{
                u_{i+1} - u_i
            }{
                h
            }

    where :math:`h` is the spacing between the nodes of the mesh and :math:`i`
    runs over the nodes indices.

    Parameters
    ----------
    node_spacing : float
        Spacing between the nodes of the mesh.
    n_nodes : int
        Amount of mesh nodes along the direction in which the differentiation
        will be carried out.

    Returns
    -------
    diff_matrix : sparse 2D matrix
        Differentiation matrix.
    """
    # Build diagonals of the sparse D matrix
    diagonal = np.ones(n_nodes) / node_spacing
    diagonals = (-diagonal, diagonal)
    diagonals_indices = (0, 1)
    # Build the sparse matrix
    shape = (n_nodes - 1, n_nodes)
    diff_matrix = sparse.spdiags(diagonals, diagonals_indices, shape[0], shape[1])
    return diff_matrix
