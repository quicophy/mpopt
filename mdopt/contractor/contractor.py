"""
The ``mdopt.contractor.contractor`` module.
======================================================
This module contains the MPS-MPO contractor functions.
"""

from typing import Union
import numpy as np
from opt_einsum import contract

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS
from mdopt.utils.utils import split_two_site_tensor


def apply_one_site_operator(tensor: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """
    Applies a one-site operator to a canonical MPS as follows::

        ----(tensor)---  ->  ---(tensor_updated)---
               |                       |
           (operator)                  |
               |                       |

    The operator can be non-unitary, however note
    that a non-unitary operator might break the canonical form.
    The operator has legs ``(pU, pD)``, where ``p`` stands for "physical", and
    ``U``, ``D`` -- for "up", "down" accordingly.

    Parameters
    ----------
    tensor : np.ndarray
        The MPS tensor to apply the operator to.
    operator : np.ndarray
        The operator to be applied.

    Returns
    -------
    tensor_updated : np.ndarray
        The updated MPS tensor.

    Raises
    ------
    ValueError
        If the MPS tensor is not three-dimensional.
    ValueError
        If the operator tensor is not two-dimensional.

    """

    if len(tensor.shape) != 3:
        raise ValueError(
            f"A valid MPS tensor must have 3 legs while the one given has {len(tensor.shape)}."
        )

    if len(operator.shape) != 2:
        raise ValueError(
            "A valid single-site operator must have 2 legs"
            f"while the one given has {len(operator.shape)}."
        )

    tensor_updated = contract("ijk, jl -> ilk", tensor, operator, optimize=[(0, 1)])

    return tensor_updated


def apply_two_site_unitary(
    lambda_0: list,
    b_1: np.ndarray,
    b_2: np.ndarray,
    unitary: np.ndarray,
    chi_max: np.int16 = 1e4,
    cut: np.float64 = 1e-6,
) -> tuple[np.ndarray]:
    """
    Applies a two-site unitary operator to a right-canonical MPS as follows::


        ---(lambda_0)---(b_1)---(b_2)---  ->  ---(b_1_updated)---(b_2_updated)---
                          |      |                     |             |
                          (unitary)                    |             |
                          |      |                     |             |

    This function uses a trick which allows performing the contraction
    without computing the inverse of any singular value matrix,
    which can introduce numerical instabilities for small singular values.
    Returns back the resulting MPS tensors in the right-canonical form.

    Unitary has legs ``(pUL pUR, pDL pDR)``, where ``p`` stands for "physical", and
    ``L``, ``R``, ``U``, ``D`` -- for "left", "right", "up", "down" accordingly.

    Parameters
    ----------
    lambda_0 : list
        A list of singular values to the left of first MPS tensor.
    b_1 : np.ndarray
        The first MPS right-isometric tensor to apply the unitary to.
    b_2 : np.ndarray
        The second MPS right-isometric tensor to apply the unitary to.
    unitary : np.ndarray
        The unitary tensor we apply.
    cut: np.float64
        Truncation tolerance for the SVD
    chi_max: np.int16
        Maximum number of singular values to keep.

    Returns
    -------
    b_1_updated : np.ndarray
        The first updated MPS right-isometric tensor.
    b_2_updated : np.ndarray
        The second updated MPS right-isometric tensor.

    Raises
    ------
    ValueError
        If the first MPS tensor is not three-dimensional.
    ValueError
        If the second MPS tensor is not three-dimensional.
    ValueError
        If the operator tensor is not four-dimensional.

    """

    if len(b_1.shape) != 3:
        raise ValueError(
            f"A valid MPS tensor must have 3 legs while the b_1 given has {len(b_1.shape)}."
        )

    if len(b_2.shape) != 3:
        raise ValueError(
            f"A valid MPS tensor must have 3 legs while the b_2 given has {len(b_1.shape)}."
        )

    if len(unitary.shape) != 4:
        raise ValueError(
            "A valid two-site operator must have 4 legs"
            f"while the one given has {len(unitary.shape)}."
        )

    two_site_tensor_with_lambda_0 = contract(
        "ij, jkl, lmn -> ikmn", np.diag(lambda_0), b_1, b_2, optimize=[(0, 1), (0, 1)]
    )
    two_site_tensor_with_lambda_0 = contract(
        "ijkl, jkmn -> imnl", two_site_tensor_with_lambda_0, unitary, optimize=[(0, 1)]
    )

    two_site_tensor_wo_lambda_0 = contract("ijk, klm", b_1, b_2, optimize=[(0, 1)])
    two_site_tensor_wo_lambda_0 = contract(
        "ijkl, jkmn -> imnl", two_site_tensor_wo_lambda_0, unitary, optimize=[(0, 1)]
    )

    _, _, b_2_updated = split_two_site_tensor(
        two_site_tensor_with_lambda_0, chi_max=chi_max, cut=cut, renormalise=False
    )
    b_1_updated = contract(
        "ijkl, mkl -> ijm",
        two_site_tensor_wo_lambda_0,
        np.conjugate(b_2_updated),
        optimize=[(0, 1)],
    )

    return (b_1_updated, b_2_updated)


def mps_mpo_contract(
    mps: Union[ExplicitMPS, CanonicalMPS],
    mpo: list[np.ndarray],
    start_site: np.int16 = 0,
    renormalise: bool = False,
    chi_max: np.int64 = 1e4,
    cut: np.float64 = 1e-6,
    inplace: bool = False,
) -> CanonicalMPS:
    """
    Applies an MPO to an MPS.

    Applies an operator (not necessarily unitary) in the MPO format
    to a canonical MPS with the orthogonality centre at site ``start_site``
    while optionally renormalising singular values at each bond.
    Returning the updated MPS in the canonical form.

    Note, this algorithm goes from left to right.
    In order to run from right to left, reverse both the MPS and the MPO manually.

    The initial configuration looks as follows with
    ``---O---`` depicting the orthogonality centre::

        ...---( )---O----( )---( )---...---( )---...
                    |     |     |           |
                   [ ]---[ ]---[ ]---...---[ ]
                    |     |     |           |

    The contraction proceeds as described in the supplementary notes.

    Parameters
    ----------
    mps : CanonicalMPS or ExplicitMPS
        The initial MPS.
    mpo : list[np.ndarray]
        MPO as a list of tensors, where each tensor is corresponding to
        an operator applied at a certain site.
        The operators should be ordered in correspondence with the sites.
        According to our convention, each operator has legs (vL, vR, pU, pD),
        where v stands for "virtual", p -- for "physical",
        and L, R, U, D stand for "left", "right", "up", "down".
    start_site : np.int16
        Index of the starting site.
    renormalise : bool
        Whether to renormalise the singular values after each contraction
        involving two neigbouring MPS sites.
    chi_max : np.int16
        Maximum bond dimension to keep.
    cut: np.float64
        Truncation tolerance for the SVD
    inplace : bool
        Whether to modify the starting MPS or create a new one.

    Returns
    -------
    mps : CanonicalMPS
        The updated MPS in the canonical form.

    Raises
    ------
    ValueError
        If any of the MPO tensors is not four-dimensional.
    ValueError
        If the length of the MPO and the starting site doesn't correspond
        to the number of sites of the MPS.

    """

    if not inplace:
        mps = mps.copy()
    if isinstance(mps, ExplicitMPS):
        mps = mps.mixed_canonical(start_site)
    if mps.orth_centre != start_site:
        mps = mps.move_orth_centre(start_site)

    for i, tensor in enumerate(mpo):
        if len(tensor.shape) != 4:
            raise ValueError(
                f"A valid MPO tensor must have 4 legs while tensor {i} has {len(tensor.shape)}."
            )

    if start_site + len(mpo) > len(mps):
        raise ValueError(
            "The length of the MPO should correspond to the number of sites where it is applied, "
            f"given MPO of length {len(mpo)} and the starting site {start_site}, "
            f"while the MPS has length {len(mps)}."
        )

    orth_centre_index = start_site

    two_site_mps_mpo_tensor = contract(
        "ijk, klm, nojp, oqlr -> iprqm",
        mps.tensors[start_site],
        mps.tensors[start_site + 1],
        mpo[0],
        mpo[1],
        optimize=[(0, 1), (1, 2), (0, 1)],
    ).reshape(
        (
            mps.tensors[start_site].shape[0],
            mpo[0].shape[3],
            mpo[1].shape[3],
            mps.tensors[start_site + 1].shape[2] * mpo[1].shape[1],
        )
    )
    for i in range(len(mpo) - 2):

        mps.tensors[orth_centre_index], singular_values, b_r = split_two_site_tensor(
            two_site_mps_mpo_tensor, chi_max=chi_max, cut=cut, renormalise=renormalise
        )

        orth_centre_index += 1
        mps.orth_centre = orth_centre_index

        mps.tensors[orth_centre_index] = contract(
            "ij, jkl -> ikl", np.diag(singular_values), b_r, optimize=[(0, 1)]
        ).reshape(
            (
                len(singular_values),
                mpo[i + 1].shape[3],
                mpo[i + 1].shape[1],
                mps.tensors[orth_centre_index + 1].shape[0],
            )
        )

        two_site_mps_mpo_tensor = contract(
            "ijkl, lmn, komp -> ijpon",
            mps.tensors[orth_centre_index],
            mps.tensors[orth_centre_index + 1],
            mpo[i + 2],
            optimize=[(0, 1), (0, 1)],
        ).reshape(
            (
                len(singular_values),
                mps.tensors[orth_centre_index].shape[1],
                mpo[i + 2].shape[3],
                mps.tensors[orth_centre_index + 1].shape[2] * mpo[i + 2].shape[1],
            )
        )

    mps.tensors[orth_centre_index], singular_values, b_r = split_two_site_tensor(
        two_site_mps_mpo_tensor, chi_max=chi_max, cut=cut, renormalise=renormalise
    )

    mps.tensors[orth_centre_index + 1] = contract(
        "ij, jkl -> ikl", np.diag(singular_values), b_r, optimize=[(0, 1)]
    )

    return mps
