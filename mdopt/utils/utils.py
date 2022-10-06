"""This module contains miscellaneous utilities."""

import numpy as np
import scipy
from opt_einsum import contract


def svd(
    mat: np.ndarray,
    cut: np.float64 = 1e-6,
    chi_max: np.int16 = 1e4,
    renormalise: bool = False,
) -> tuple[np.ndarray]:
    """
    Performs the Singular Value Decomposition with different features.

    Parameters
    ----------
    mat : np.ndarray
        Matrix provided as a tensor with 2 dimensions.
    cut : float
        Truncation tolerance for the SVD
    chi_max : np.int16
        Maximum number of singular values to keep.
    renormalise : bool
        Whether to renormalise the singular value spectrum after the cut.

    Returns
    -------
    u_l : np.ndarray
        Unitary matrix having left singular vectors as columns.
    singular_values : np.ndarray
        The singular values, sorted in non-increasing order.
    v_r : np.ndarray
        Unitary matrix having right singular vectors as rows.

    Raises
    ------
    ValueError
        If the `np.ndarray` provided is not two-dimensional.
    """

    if len(mat.shape) != 2:
        raise ValueError(
            f"A valid matrix must have 2 dimensions while the one given has {len(mat.shape)}."
        )

    try:
        #this use lapack routine gesdd (quicker but may not converge)
        u_l, singular_values, v_r = np.linalg.svd(
            mat, full_matrices=False, compute_uv=True, hermitian=False
        )
    except:
        #Use lapack routine gesvd (slower but more stable)
        u_l, singular_values, v_r = scipy.linalg.svd(
-           mat, full_matrices=False, compute_uv=True, lapack_driver="gesvd"
        )
    mat_norm2 = np.sum(singular_values**2)
    approx_norm2 = np.cumsum(singular_values**2)
    i = np.sum(approx_norm2 <= mat_norm2*(1-cut)**2)
    if i+1 < chi_max:
        u_l, singular_values, v_r = u_l[:, 0:i+1], singular_values[:i+1], v_r[0:i+1]
    elif chi_max < len(singular_values):
        u_l, singular_values, v_r = u_l[:, 0:chi_max], singular_values[:chi_max], v_r[0:chi_max]

    if renormalise:
        singular_values /= np.linalg.norm(singular_values)

    return u_l, singular_values, v_r


def create_sparse_matrix(mat, cutoff=1e-10, f="csr"):
    mat_max = 1/np.max(mat)
    i,j = np.argwhere(abs(mat_max * mat) > cutoff).transpose()
    if f == "csr":
        mat_sparse = scipy.sparse.csr_array((mat[i,j], (i,j)), shape=mat.shape)
    if f == "csc":
        mat_sparse = scipy.sparse.csc_array((mat[i,j], (i,j)), shape=mat.shape)
    return mat_sparse


def kron_tensors(
    tensor_1: np.ndarray,
    tensor_2: np.ndarray,
    conjugate_second: bool = False,
    merge_physicals: bool = True,
) -> np.ndarray:
    """
    Computes a kronecker product of 2 MPS tensors.

    Parameters
    ----------
    tensor_1 : np.ndarray
        The first tensor of the product.
    tensor_2 : np.ndarray
        The second tensor of the product.
    conjugate_second : bool
        Whether to complex-conjugate the second tensor.
    merge_physicals : bool
        Whether to merge physical indices.

    Returns
    -------
    product
        The resulting kronecker product.

    Raises
    ------
    ValueError
        If the first MPS tensor is not three-dimensional.
    ValueError
        If the second MPS tensor is not three-dimensional.

    Notes
    -----
    This function acts according to the following diagram::

           tensor_2
              i
              |                      i
        j ---( )--- k                |
                        -->    jl---( )---kn
        l ---( )--- n                |
              |                      m
              m
           tensor_1

    The legs of the resulting tensor are indexed as ``(jl, m, i, kn)``.
    Indices `i` and `m` can be merged if `merge_physicals=True`.
    """

    if len(tensor_1.shape) != 3:
        raise ValueError(
            f"The number of dimensions of the first tensor is {len(tensor_1.shape)},"
            "but the number of dimensions expected is 3."
        )
    if len(tensor_2.shape) != 3:
        raise ValueError(
            f"The number of dimensions of the second tensor is {len(tensor_2.shape)},"
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


def split_two_site_tensor(
    tensor: np.ndarray,
    chi_max: np.int16 = 1e4,
    cut: np.float64 = 1e-6,
    renormalise: bool = False,
) -> tuple:
    """
    Split a two-site MPS tensor according to the following diagram::

       i ---(tensor)--- l     ->    i ---(A)---diag(S)---(B)--- l
             |   |                         |               |
             j   k                         j               k

    Parameters
    ----------
    tensor : np.ndarray
        Two-site tensor ``(i, j, k, l)``.
    chi_max : np.int16
        Maximum number of singular values to keep.
    cut : np.float64
        Truncation tolerance for tensor splitting, i.e. find the rank k such as:
        || tensor - Ak * diag(Sk) * Bk || < cut

    Returns
    -------
    a_l : np.ndarray
        Left isometry ``(i, j, m)``.
    singular_values : np.ndarray
        List of singular values.
    b_r : np.ndarray
        Right isometry ``(m, k, l)``.

    Raises
    ------
    ValueError
        If the tensor is not four-dimensional.
    """

    if len(tensor.shape) != 4:
        raise ValueError(
            "A valid two-site tensor must have 4 legs"
            f"while the one given has {len(tensor.shape)}."
        )

    chi_v_l, d_l, d_r, chi_v_r = tensor.shape
    tensor = tensor.reshape((chi_v_l * d_l, d_r * chi_v_r))

    u_l, singular_values, v_r = svd(
        tensor, cut=cut, chi_max=chi_max, renormalise=renormalise
    )

    chi_v_cut = len(singular_values)
    a_l = u_l.reshape((chi_v_l, d_l, chi_v_cut))
    b_r = v_r.reshape((chi_v_cut, d_r, chi_v_r))

    return a_l, singular_values, b_r


def create_random_mpo(
    num_sites: np.int16,
    bond_dimensions: list[np.int16],
    phys_dim: np.int16,
    which: str = "uniform",
) -> list[np.ndarray]:
    """
    Creates a random complex-valued Matrix Product Operator.

    Parameters
    ----------
    num_sites : np.int16
        The number of sites for the MPO.
        This will be equal to the number of tensors.
    bond_dimensions : list[np.int16]
        A list of bond dimensions.
    phys_dim : np.int16
        Physical dimension of the tensors.
    which : str
        Specifies the distribution from which
        the matrix elements are being taken.
        Options: "uniform", "normal", "randint".

    Returns
    -------
    mpo : list[np.ndarray]
        The resulting MPO.

    Notes
    -----
    The ``bond_dimensions`` argument should be given as a list of right virtual dimensions
    without the last trivial virtual dimension. Thus, the length of the list is ``num_sites - 1``.

    The distributions available: uniform(0,1), normal, random integer {0,1}.

    Each tensor in the MPO list has legs ``(vL, vR, pU, pD)``, where ``v`` stands for "virtual",
    ``p`` -- for "physical", and ``L, R, U, D`` -- for "left", "right", "up", "down" accordingly.
    """

    bonds = [[dim, dim] for dim in bond_dimensions]
    bonds.append([1])
    bonds.insert(0, [1])
    bonds = [item for sublist in bonds for item in sublist]

    shapes = [
        (bonds[i], bonds[i + 1], phys_dim, phys_dim)
        for i in range(0, len(bonds) - 1, 2)
    ]

    if which == "uniform":
        mpo = [
            np.random.uniform(size=shapes[i]) + 1j * np.random.uniform(size=shapes[i])
            for i in range(num_sites)
        ]

    if which == "normal":
        mpo = [
            np.random.normal(size=shapes[i]) + 1j * np.random.normal(size=shapes[i])
            for i in range(num_sites)
        ]

    if which == "randint":
        mpo = [
            np.random.randint(0, 2, size=shapes[i])
            + 1j * np.random.randint(0, 2, size=shapes[i])
            for i in range(num_sites)
        ]

    return mpo


def mpo_to_matrix(
    mpo: list[np.ndarray], interlace: bool = False, group: bool = False
) -> np.ndarray:
    """
    Creates a matrix from an MPO.

    Parameters
    ----------
    mpo : list[np.ndarray]
        The MPO to convert to a matrix.
    interlace : bool
        Whether to interlace the matrix' legs or not.
    group : bool
        Whether to group the matrix' legs or not, see the notes.
        Grouping means merging all the up legs into one leg and the same for the down legs.

    Returns
    -------
    matrix : np.ndarray
        The resulting matrix.

    Raises
    ------
    ValueError
        If any of the MPO tensors is not four-dimensional.

    Notes
    -----
    If ``interlace==True``, the matrix' legs will go as
    ``(p0U, p0D, p1U, p1D, ...)``, which means
    physical legs sticking up and down with the site number.
    If ``interlace==False``, the matrix' legs will go as
    ``(p0D, p1D, ..., p0U, p1U, ...)``, which means listing first
    all physical legs sticking down with the site number,
    and then all physical legs sticking up.
    This is done to adjust the matrix to the ``@`` numpy-native matrix-vector multiplication.
    Note, grouping (if wanted) is being done after the interlacing.

    An example of a matrix with ungrouped legs on three sites::

          p0U p1U p2U
         __|___|___|__
        |            |
        |____________|
           |   |   |
          p0D p1D p2D

    Each tensor in the MPO list has legs ``(vL, vR, pU, pD)``, where v stands for "virtual",
    p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.
    Warning: will cause memory overflow for number of sites > ~20.
    """

    for i, tensor in enumerate(mpo):
        if len(tensor.shape) != 4:
            raise ValueError(
                f"A valid MPO tensor must have 4 legs while tensor {i} has {len(tensor.shape)}."
            )

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


def mpo_from_matrix(
    matrix: np.ndarray,
    num_sites: np.int16,
    interlaced: bool = True,
    phys_dim: np.int16 = 2,
    chi_max: np.int16 = 1e4,
) -> list[np.ndarray]:
    """
    Creates an MPO from a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to convert to an MPO.
        Can be given with either physical legs grouped together or merged (see notes).
    num_sites : np.int16
        The number of sites in the MPO.
    interlaced : bool
        Whether the matrix legs are interlaced or not.
    phys_dim : np.int16
        Local dimension of the physical legs.
    chi_max : np.int16
        Maximum bond dimension allowed in the MPO.

    Returns
    -------
    mpo : list[np.ndarray]
        The resulting Matrix Product Operator.

    Raises
    ------
    ValueError
        If the matrix' shape does not correspond to ``phys_dim`` and ``num_sites``.

    Notes
    -----
    If ``interlaced==True``, the matrix' legs are considered to go as
    ``(p0U, p0D, p1U, p1D, ...)``, which means physical legs sticking up and down with the site number.
    If ``interlaced==False``, the matrix' legs are considered to go as
    ``(p0D, p1D, ..., p0U, p1U, ...)``, which means listing first all physical legs sticking down
    with the site number, and then all physical legs sticking up.
    This is done to adjust the matrix to the ``@`` numpy-native matrix-vector multiplication.

    An example of a matrix with ungrouped legs on three sites::

          p0U p1U p2U
         __|___|___|__
        |            |
        |____________|
           |   |   |
          p0D p1D p2D

    Each tensor in the ``mpo`` list will have legs ``(vL, vR, pU, pD)``, where ``v`` stands for "virtual",
    ``p`` -- for "physical", and ``L, R, U, D`` -- for "left", "right", "up", "down" accordingly.
    """

    hilbert_space_dim = phys_dim**num_sites
    phys_dims = [phys_dim] * num_sites * 2
    mps_dim = phys_dim**2

    # Checking the matrix has correct shapes for
    # a given physical dimension and number of sites.
    if (matrix.shape != tuple([hilbert_space_dim] * 2)) and (
        matrix.shape != tuple(phys_dims)
    ):
        raise ValueError(
            f"The matrix' shape should be either {tuple([hilbert_space_dim] * 2)}, "
            f"or {tuple(phys_dims)}, instead, the matrix given has shape {matrix.shape}."
        )

    # Copying the matrix not to change the original one inplace.
    matrix = matrix.copy()

    # Reshaping the matrix.
    if matrix.shape == tuple([hilbert_space_dim] * 2):
        matrix = matrix.reshape(tuple(phys_dims))

    # Dealing with possible interlacing.
    if not interlaced:
        correct_order = list([i + num_sites, i] for i in range(num_sites))
        correct_order = [item for sublist in correct_order for item in sublist]
        matrix = matrix.transpose(correct_order)

    # Treating the MPO as an MPS with squared physical dimensions.
    mpo = []
    matrix = matrix.reshape((-1, mps_dim))
    matrix, singular_values, v_r = svd(matrix, chi_max=chi_max, renormalise=False)
    v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
    mpo.append(np.expand_dims(v_r, -1))
    while matrix.shape[0] >= mps_dim:
        matrix = np.matmul(matrix, np.diag(np.sqrt(singular_values)))
        bond_dim = matrix.shape[-1]
        matrix = matrix.reshape((-1, mps_dim * bond_dim))
        matrix, singular_values, v_r = svd(matrix, chi_max=chi_max, renormalise=False)
        v_r = np.matmul(np.diag(np.sqrt(singular_values)), v_r)
        v_r = v_r.reshape((-1, mps_dim, bond_dim))
        mpo.insert(0, v_r)

    # Contracting in the orthogonality centre.
    mpo[0] = contract(
        "ij, jk, klm",
        matrix,
        np.diag(np.sqrt(singular_values)),
        mpo[0],
        optimize=[(0, 1), (0, 1)],
    )

    # Converting the MPS back to the MPO.
    mpo = [tensor.transpose((0, 2, 1)) for tensor in mpo]
    mpo = [
        tensor.reshape((tensor.shape[0], tensor.shape[1], phys_dim, phys_dim))
        for tensor in mpo
    ]

    return mpo
