"""
This module contains relevant utilities.
"""

import numpy as np
from opt_einsum import contract
import pytaco as pt


def svd(mat, cut=1e-14, chi_max=1e5, renormalise=False):
    """
    Returns the Singular Value Decomposition of a matrix `mat`.

    Arguments:
        mat: np.array[ndim=2]
            Matrix given as a `np.array` with 2 dimensions.
        cut: float
            Singular values smaller than this will be discarded.
        chi_max: int
            Maximum number of singular values to keep.
        renormalise: bool
            Renormalisation of the singular value spectrum.

    Returns:
        u_l: ndarray
            Unitary matrix having left singular vectors as columns.
        singular_values: ndarray
            The singular values, sorted in non-increasing order.
        v_r: ndarray
            Unitary matrix having right singular vectors as rows.
    """

    u_l, singular_values, v_r = np.linalg.svd(mat, full_matrices=False)

    max_num = min(chi_max, np.sum(singular_values > cut))
    ind = np.argsort(singular_values)[::-1][:max_num]

    u_l, singular_values, v_r = u_l[:, ind], singular_values[ind], v_r[ind, :]

    if renormalise:
        singular_values /= np.linalg.norm(singular_values)

    return u_l, singular_values, v_r


def kron_tensors(tensor_1, tensor_2, conjugate_second=False, merge_physicals=True):
    """
    An utility function which is used to compute
    different versions of a kronecker product of 2 MPS tensors.
    This function acts according to the following cartoon.
    Note, that indices `i` and `m` can be merged if `merge_physicals=True`.

          tensor_2

              i
              |                      i
        j ---()--- k                 |
                        -->    jl---()---kn = np.array((jl, m, i, kn))
        l ---()--- n                |
             |                      m
             m

          tensor_1

    Arguments:
        tensor_1: np.array[ndim=3]
            The first tensor of the product.
        tensor_2: np.array[ndim=3]
            The second tensor of the product.
        conjugate_second: bool
            Whether to complex-conjugate the second tensor.
        merge_physicals: bool
            Whether to merge physical indices.
    """

    if len(tensor_1.shape) != len(tensor_2.shape):
        raise ValueError("The tensors must have equal numbers of dimensions.")

    if len(tensor_1.shape) != 3:
        raise ValueError(
            f"The number of dimensions given was ({len(tensor_1.shape)}), "
            "but the number of dimensions expected is 3."
        )

    if conjugate_second:
        product = np.kron(tensor_1, np.conjugate(tensor_2))
    else:
        product = np.kron(tensor_1, tensor_2)

    if not merge_physicals:
        return product.reshape(
            (
                tensor_1.shape[0] * tensor_2.shape[0],
                tensor_1.shape[1],
                tensor_2.shape[1],
                tensor_1.shape[2] * tensor_2.shape[2],
            )
        )

    return product


def split_two_site_tensor(theta, chi_max=1e5, cut=1e-14, renormalise=False):
    """
    Split a two-site MPS tensor as follows:
          vL --(theta)-- vR     ->    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Parameters:
        theta : np.array[ndim=4]
            Two-site tensor, with legs vL, i, j, vR.
        chi_max : int
            Maximum number of singular values to keep.
        eps : float
            Discard any singular values smaller than eps.

    Returns:
        a_l : np.array[ndim=3]
            Left isometry on site i, with legs vL, i, vC.
        singular_values : np.array[ndim=1]
            List of singular values.
        b_r : np.array[ndim=3]
            Right isometry on site j, with legs vC, j, vR.
    """

    # merge the legs to form a matrix to feed into the svd
    chi_v_l, d_l, d_r, chi_v_r = theta.shape
    theta = theta.reshape((chi_v_l * d_l, d_r * chi_v_r))

    # do a trimmed svd
    u_l, singular_values, v_r = svd(
        theta, cut=cut, chi_max=chi_max, renormalise=renormalise
    )

    # split legs of u_l and v_r
    chi_v_cut = len(singular_values)
    a_l = u_l.reshape((chi_v_l, d_l, chi_v_cut))
    b_r = v_r.reshape((chi_v_cut, d_r, chi_v_r))

    return a_l, singular_values, b_r


def create_random_mpo(num_sites, bond_dims, phys_dim=2, which="uniform"):
    """
    A function which creates a random complex-valued Matrix Product Operator.

    The `bond_dims` argument should be given as a list of right virtual dimensions
    without the last trivial virtual dimension. It means, the length of the list is `num_sites - 1`.

    The distributions available in the function: uniform(0,1), normal, random integer {0,1}.

    Each tensor in the MPO list has legs (vL, vR, pU, pD), where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.

    Arguments:
        num_sites: int
            The number of sites for the MPO.
            This will be equal to the number of tensors.
        bond_dims: list
            A list of bond dimensions.
        phys_dim: int
            Physical dimension of the tensors.
        which: str
            Specifies the distribution from which
            the matrix elements are being taken.
            Options: "uniform", "normal", "randint".
    """

    bonds = [[dim, dim] for dim in bond_dims]
    bonds.append([1])
    bonds.insert(0, [1])
    bonds = [item for sublist in bonds for item in sublist]

    shapes = [
        (bonds[i], bonds[i + 1], phys_dim, phys_dim)
        for i in range(0, len(bonds) - 1, 2)
    ]

    if which == "uniform":
        mps = [
            np.random.uniform(size=shapes[i]) + 1j * np.random.uniform(size=shapes[i])
            for i in range(num_sites)
        ]

    if which == "normal":
        mps = [
            np.random.normal(size=shapes[i]) + 1j * np.random.normal(size=shapes[i])
            for i in range(num_sites)
        ]

    if which == "randint":
        mps = [
            np.random.randint(0, 2, size=shapes[i])
            + 1j * np.random.randint(0, 2, size=shapes[i])
            for i in range(num_sites)
        ]

    return mps


def mpo_to_matrix(mpo, interlace=False, group=False):
    """
    A utility function allowing the creation of a matrix from an MPO.

    If `interlace` is `True`, the matrix' legs will go as
    (p0U, p0D, p1U, p1D, ...), which means
    physical legs sticking up and down with the site number.
    If `interlace` is `False`, the matrix' legs will go as
    (p0D, p1D, ..., p0U, p1U, ...), which means listing first
    all physical legs sticking down with the site number,
    and then all physical legs sticking up.
    This is done to adjust the matrix to the @ matrix-vector multiplication.
    Note, grouping (if wanted) is being done after the interlacing.

        p0U p1U p2U
     ___|___|___|___
    |              | -- example of a 3-site matrix with legs ungrouped.
    |______________|
       |   |   |
      p0D p1D p2D

    Each tensor in the MPO list has legs (vL, vR, pU, pD), where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.
    Warning: will cause memory overflow for number of sites > ~20.

    Arguments:
        mpo: List of np.ndarray[ndim=4]
            The MPO to convert to a matrix.
        interlace: bool
            Whether to interlace the matrix' legs or not.
        group: bool
            Whether to group the matrix' legs or not.
            Grouping means merging all the up legs into one leg and the same for the down legs.
    """

    phys_dim = mpo[0].shape[2]
    num_sites = len(mpo)
    matrix = (
        np.tensordot(mpo[0], mpo[1], (1, 0))
        .transpose((0, 3, 1, 2, 4, 5))
        .reshape((-1, mpo[1].shape[1], phys_dim**2, phys_dim**2))
    )

    for i in range(num_sites - 2):

        matrix = (
            np.tensordot(matrix, mpo[i + 2], (1, 0))
            .transpose((0, 3, 1, 2, 4, 5))
            .reshape(
                (-1, mpo[i + 2].shape[1], phys_dim ** (i + 3), phys_dim ** (i + 3))
            )
        )

    matrix = matrix.reshape(([phys_dim] * num_sites * 2))

    if not interlace:
        order = list(range(1, 2 * num_sites, 2)) + list(range(0, 2 * num_sites, 2))
        matrix = matrix.transpose(order)

    if group:
        matrix = matrix.reshape((phys_dim**num_sites, phys_dim**num_sites))

    return matrix


def mpo_from_matrix(matrix, num_sites, interlaced=True, phys_dim=2, chi_max=1e5):
    """
    A utility function allowing the creation of a Matrix Product Operator from a matrix.

    If `interlaced` is `True`, the matrix' legs are considered to go as
    (p0U, p0D, p1U, p1D, ...), which means physical legs sticking up and down with the site number.
    If `interlaced` is `False`, the matrix' legs are considered to go as
    (p0D, p1D, ..., p0U, p1U, ...), which means listing first all physical legs sticking down
    with the site number, and then all physical legs sticking up.
    This is done to adjust the matrix to the @ matrix-vector multiplication.

        p0U p1U p2U
     ___|___|___|___
    |              | -- example of a 3-site matrix
    |______________|
       |   |   |
      p0D p1D p2D

    Each tensor in the `mpo` list will have legs (vL, vR, pU, pD), where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.

    Arguments:
        matrix: np.array
            The matrix to convert to an MPO.
            Can be given with either physical legs grouped together or merged.
        interlaced: bool
            Whether the matrix legs are interlaced or not.
        num_sites: int
            The number of sites in the MPO.
        phys_dim: int
            Local dimension of the physical legs.
        chi_max: int
            Maximum bond dimension allowed in the MPO.
    """

    # Defining some dimensions we will need further
    hilbert_space_dim = phys_dim**num_sites
    phys_dims = [phys_dim] * num_sites * 2
    mps_dim = phys_dim**2

    # Checking the matrix has correct shapes for
    # a given physical dimension and number of sites
    if (matrix.shape != tuple([hilbert_space_dim] * 2)) and (
        matrix.shape != tuple(phys_dims)
    ):
        raise ValueError(
            f"The matrix' shape should be either {tuple([hilbert_space_dim] * 2)}, "
            f"or {tuple(phys_dims)}, instead, the matrix given has shape {matrix.shape}."
        )

    # Copying the matrix not to change the original one inplace
    mat = matrix.copy()

    # Reshaping the matrix
    if matrix.shape == tuple([hilbert_space_dim] * 2):
        mat = mat.reshape(tuple(phys_dims))

    # Dealing with possible interlacing
    if not interlaced:
        correct_order = list([i + num_sites, i] for i in range(num_sites))
        correct_order = [item for sublist in correct_order for item in sublist]
        mat = mat.transpose(correct_order)

    # Treating the MPO as an MPS with squared physical dimensions
    mpo = []
    mat = mat.reshape((-1, mps_dim))
    mat, singular_values, v_r = svd(mat, chi_max=chi_max, renormalise=False)
    v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
    mpo.append(np.expand_dims(v_r, -1))
    while mat.shape[0] >= mps_dim:
        mat = np.matmul(mat, np.diag(np.sqrt(singular_values)))
        bond_dim = mat.shape[-1]
        mat = mat.reshape((-1, mps_dim * bond_dim))
        mat, singular_values, v_r = svd(mat, chi_max=chi_max, renormalise=False)
        v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
        v_r = v_r.reshape((-1, mps_dim, bond_dim))
        mpo.insert(0, v_r)

    # Contracting in the orthogonality centre
    mpo[0] = contract(
        "ij, jk, klm",
        mat,
        np.diag(np.sqrt(singular_values)),
        mpo[0],
        optimize=[(0, 1), (0, 1)],
    )

    # Converting the MPS back to the MPO
    mpo = [tensor.transpose((0, 2, 1)) for tensor in mpo]
    mpo = [
        tensor.reshape((tensor.shape[0], tensor.shape[1], phys_dim, phys_dim))
        for tensor in mpo
    ]
    return mpo

def numpy_dense_to_pytaco_sparse(A, rel_cutoff=1e-6, abs_cutoff=1e-12, dtype=None):
    """
    Create a sparse pytaco tensor from a numpy dense tensor.
    Also remove tensor components below abs_cutoff and below rel_cutoff time
    the highest tensor component in absolute value.
    Removal of small component is carried on the input tensor
    Tensor structure is reallocated.
    """
    if isinstance(A, pt.tensor): 
        print("Tensor is already a pytaco tensor")
        return A
    # TODO: it copy full array to remove after the 0, maybe I can make something smarter
    #zero small components
    #rel
    A[np.abs(A) < rel_cutoff * np.max(A)] = 0
    #abs
    A[A < abs_cutoff] = 0
    #conversion
    A_ = pt.from_array(A)
    #retrieve desired precision
    if dtype == "double":
        dtype = pt.float64
    elif dtype == "float":
        dtype = pt.float32
    #remove explicit 0
    A_ = pt.remove_explicit_zeros(
        A_,
        [pt.compressed for _ in A.shape],
        dtype
    ) 
    return A_
