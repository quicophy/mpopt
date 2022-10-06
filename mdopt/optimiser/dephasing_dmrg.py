"""This module contains the :class:`DephasingDMRG` and the :class:`EffectiveDensityOperator`
classes.

This algorithm's main feature is that it restricts the target-state search to
the computational basis states domain.
In particular, we use it to find the main component of a Matrix Density Product Operator (MDPO),
i.e., a computational basis state contributing the largest amplitude.

In our notation, MDPO for ``n`` sites denotes the following object::

         |      |               |       |
         |      |               |       |
    ----(0*)---(1*)--- ... ---(n-2*)--(n-1*)---
    ----(0)----(1)---- ... ---(n-2)---(n-1)----
         |      |               |       |
         |      |               |       |

An MDPO is formed by an MPS and its complex-conjugated version.
The main idea is to find the main component of this object without
performing the kronecker product explicitly.
"""


from typing import Union
import numpy as np
import scipy.sparse
from opt_einsum import contract
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from mdopt.mps.canonical import CanonicalMPS
from mdopt.mps.explicit import ExplicitMPS
from mdopt.utils.utils import split_two_site_tensor


class EffectiveDensityOperator(scipy.sparse.linalg.LinearOperator):
    """
    Class to store an effective two-site density operator.

    To take more advantage of the ``scipy.sparse.linalg`` module, we make a special class
    for local effective density operators extending the analogy from local effective operators.
    It allows us to compute eigenvectors more effeciently.

    The diagram displaying the contraction can be found in the supplementary notes.
    """

    def __init__(
        self,
        left_environment: np.ndarray,
        mps_target_1: np.ndarray,
        mps_target_2: np.ndarray,
        right_environment: np.ndarray,
    ):
        """
        Initialise an effective dephased density operator tensor network.

        Parameters
        ----------
        left_environment : np.ndarray
            The left environment for the effective dephased density operator.
        mps_target_1 : np.ndarray
            The left target matrix product state tensor.
        mps_target_2 : np.ndarray
            The right target matrix product state tensor.
        right_environment : np.ndarray
            The right environment for the effective dephased density operator.

        Raises
        ------
        ValueError
            If the left environment tensor is not four-dimensional.
        ValueError
            If the first target MPS tensor is not three-dimensional.
        ValueError
            If the second target MPS tensor is not three-dimensional.
        ValueError
            If the right environment tensor is not four-dimensional.
        """
        if len(left_environment.shape) != 4:
            raise ValueError(
                "A valid left environment tensor must have 4 legs"
                f"while the one given has {len(left_environment.shape)}."
            )
        if len(mps_target_1.shape) != 3:
            raise ValueError(
                "A valid target MPS tensor must have 3 legs"
                f"while the one given has {len(mps_target_1.shape)}."
            )
        if len(mps_target_2.shape) != 3:
            raise ValueError(
                "A valid target MPS tensor must have 3 legs"
                f"while the one given has {len(mps_target_1.shape)}."
            )
        if len(right_environment.shape) != 4:
            raise ValueError(
                "A valid right environment tensor must have 4 legs"
                f"while the one given has {len(right_environment.shape)}."
            )

        self.left_environment = left_environment
        self.right_environment = right_environment
        self.mps_target_1 = mps_target_1
        self.mps_target_2 = mps_target_2
        chi_1, chi_2 = (
            left_environment.shape[3],
            right_environment.shape[3],
        )
        d_1, d_2 = (
            mps_target_1.shape[1],
            mps_target_2.shape[1],
        )
        self.x_shape = (chi_1, d_1, d_2, chi_2)
        self.shape = (chi_1 * d_1 * d_2 * chi_2, chi_1 * d_1 * d_2 * chi_2)
        self.dtype = mps_target_1.dtype
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Performs matrix-vector multiplication.

        Computes ``effective_density_operator * |x> = |x'>``.
        This function is used by :func:`scipy.sparse.linalg.eigsh` to diagonalise
        the effective density operator with the Lanczos method, withouth generating the full matrix.

        Parameters
        ----------
        x : np.array
            The two-site tensor to be acted on by an effective density operator.
        """

        two_site_tensor = np.reshape(x, self.x_shape)

        if len(two_site_tensor.shape) != 4:
            raise ValueError(
                f"A valid two-site tensor must have 4 legs"
                f"while the one given has {len(two_site_tensor.shape)}."
            )

        copy_tensor = np.fromfunction(
            lambda i, j, k: np.logical_and(i == j, j == k), (2, 2, 2), dtype=np.int16
        )

        einsum_string = (
            "ustw, ailu, ifj, jhk, bef, cgh, lom, mpn, eso, gtp, dknw -> abcd"
        )
        two_site_tensor = contract(
            einsum_string,
            two_site_tensor,
            self.left_environment,
            np.conjugate(self.mps_target_1),
            np.conjugate(self.mps_target_2),
            copy_tensor,
            copy_tensor,
            self.mps_target_1,
            self.mps_target_2,
            copy_tensor,
            copy_tensor,
            self.right_environment,
            optimize=[
                (0, 8),
                (0, 1),
                (0, 6),
                (0, 5),
                (1, 2),
                (2, 3),
                (3, 4),
                (2, 3),
                (1, 2),
                (0, 1),
            ],
        )

        return np.reshape(two_site_tensor, self.shape[0])


class DephasingDMRG:
    """
    Class holding the Dephasing Density Matrix Renormalisation Group algorithm with two-site updates
    for a finite-size system with open-boundary conditions.

    Attributes
    ----------
    mps
        MPS serving as a current approximation of the target state.
    mps_target
        The target MPS in the right-canonical form.
        This MPS is used to construct the dephased MDPO.
    chi_max
        The highest bond dimension of an MPS allowed.
    mode
        Available options:
            | ``LM`` : Largest (in magnitude) eigenvalues.
            | ``SM`` : Smallest (in magnitude) eigenvalues.
            | ``LA`` : Largest (algebraic) eigenvalues.
            | ``SA`` : Smallest (algebraic) eigenvalues.
    cut
        The lower boundary of the spectrum, i.e., all
        the singular values smaller than that will be discarded.
    silent
        Whether to show/hide the progress bar.
    """

    def __init__(
        self,
        mps: Union[ExplicitMPS, CanonicalMPS],
        mps_target: Union[ExplicitMPS, CanonicalMPS],
        chi_max: np.int16 = 1e4,
        cut: np.float64 = 1e-6,
        mode: str = "SA",
        silent: bool = False,
        copy: bool = True,
    ):
        if len(mps) != len(mps_target):
            raise ValueError(
                f"The MPS has length {len(mps)},"
                f"the target MPS has length {len(mps_target)},"
                "but the lengths should be equal."
            )
        if copy:
            self.mps = mps.copy()
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.right_canonical()
        self.mps = mps
        self.left_environments = [None] * len(mps)
        self.right_environments = [None] * len(mps)
        mps_target = mps_target.right_canonical()
        self.mps_target = mps_target
        self.chi_max = chi_max
        self.cut = cut
        self.mode = mode
        self.silent = silent

        start_bond_dim = self.mps_target.tensors[0].shape[0]
        chi = mps.tensors[0].shape[0]
        left_environment = np.zeros(
            [chi, start_bond_dim, start_bond_dim, chi], dtype=np.float64
        )
        right_environment = np.zeros(
            [chi, start_bond_dim, start_bond_dim, chi], dtype=np.float64
        )
        left_environment[:, 0, 0, :] = np.eye(chi, dtype=np.float64)
        right_environment[:, start_bond_dim - 1, start_bond_dim - 1, :] = np.eye(
            chi, dtype=np.float64
        )
        self.left_environments[0] = left_environment
        self.right_environments[-1] = right_environment
        for i in reversed(range(1, len(mps))):
            self.update_right_environment(i)

    def sweep(self):
        """
        One Dephasing DMRG sweep.

        A method performing one Dephasing DMRG sweep, which consists of
        two series of ``update_bond`` sweeps which go back and forth.
        """

        for i in range(self.mps.num_sites - 1):
            self.update_bond(i)

        for i in reversed(range(self.mps.num_sites - 1)):
            self.update_bond(i)

    def update_bond(self, i: np.int16):
        """
        Updates the bond between sites ``i`` and ``i+1``.
        """

        j = i + 1

        effective_density_operator = EffectiveDensityOperator(
            self.left_environments[i],
            self.mps_target.tensors[i],
            self.mps_target.tensors[j],
            self.right_environments[j],
        )

        if isinstance(self.mps, ExplicitMPS):
            initial_guess = self.mps.two_site_right_iso(i).reshape(
                effective_density_operator.shape[0]
            )
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.move_orth_centre(i)
            initial_guess = self.mps.two_site_tensor_next(i).reshape(
                effective_density_operator.shape[0]
            )

        _, eigenvectors = eigsh(
            effective_density_operator,
            k=1,
            which=self.mode,
            return_eigenvectors=True,
            v0=initial_guess,
            tol=1e-8,
        )
        x = eigenvectors[:, 0].reshape(effective_density_operator.x_shape)
        left_iso_i, singular_values_j, right_iso_j = split_two_site_tensor(
            x, chi_max=self.chi_max, cut=self.cut, renormalise=True
        )

        if isinstance(self.mps, CanonicalMPS):
            self.mps.tensors[i] = np.tensordot(
                left_iso_i, np.diag(singular_values_j), (1, 0)
            )
            self.mps.orth_centre = i
            self.mps.tensors[j] = right_iso_j

        if isinstance(self.mps, ExplicitMPS):
            self.mps.tensors[i] = np.tensordot(
                np.linalg.inv(np.diag(self.mps.singular_values[i])), left_iso_i, (1, 0)
            )
            self.mps.tensors[j] = np.tensordot(
                right_iso_j,
                np.linalg.inv(np.diag(self.mps.singular_values[j + 1])),
                (2, 0),
            )
            self.mps.singular_values[j] = singular_values_j

        self.update_left_environment(i)
        self.update_right_environment(j)

    def update_right_environment(self, i: np.int16):
        """
        Compute the ``right_environment`` right of site ``i-1``
        from the ``right_environment`` right of site ``i``.
        """

        right_environment = self.right_environments[i]

        if isinstance(self.mps, ExplicitMPS):
            right_iso = self.mps.single_site_right_iso(i)
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.move_orth_centre(i - 1)
            right_iso = self.mps.single_site_tensor(i)

        right_environment = contract(
            "ijkl, omi, pmj, qnk, rnl -> opqr",
            right_environment,
            right_iso,
            np.conjugate(self.mps_target.tensors[i]),
            self.mps_target.tensors[i],
            np.conjugate(right_iso),
            optimize=[(0, 2), (0, 1), (0, 1), (0, 1)],
        )
        self.right_environments[i - 1] = right_environment

    def update_left_environment(self, i: np.int16):
        """
        Compute the ``left_environment`` left of site ``i+1``
        from the  ``left_environment`` left of site ``i``.
        """

        left_environment = self.left_environments[i]

        if isinstance(self.mps, ExplicitMPS):
            left_iso = self.mps.single_site_left_iso(i)
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.move_orth_centre(i + 1)
            left_iso = self.mps.single_site_tensor(i)

        left_environment = contract(
            "ijkl, imo, jmp, knq, lnr -> opqr",
            left_environment,
            left_iso,
            np.conjugate(self.mps_target.tensors[i]),
            self.mps_target.tensors[i],
            np.conjugate(left_iso),
            optimize=[(0, 2), (0, 1), (0, 1), (0, 1)],
        )
        self.left_environments[i + 1] = left_environment

    def run(self, num_iter: np.int16 = 1):
        """
        Run the algorithm, i.e., run the ``sweep`` method for ``num_iter`` number of times.
        """

        for _ in tqdm(range(num_iter), disable=self.silent):
            self.sweep()
