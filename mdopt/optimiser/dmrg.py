"""This module contains the :class:`DMRG` and the :class:`EffectiveOperator` classes.
Inspired by TenPy.
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


class EffectiveOperator(scipy.sparse.linalg.LinearOperator):
    r"""Class to store an effective two-site operator.

    In order to take more advantage of the ``scipy.sparse.linalg`` module,
    we make a special class for local effective operators.
    It allows us to compute eigenvectors more effeciently.

    Such effective operator is to be diagonalised in the
    ``update_bond`` method of the :class:'DMRG' class::

        ---uL                    uR---
        |        i          j        |
        |  vL    |          |    vR  |
        (L)----(mpo_l)----(mpo_r)----(R)
        |        |          |        |
        |        k          l        |
        ---dL                    dR---

    In our convention, the legs of left/right environments (tensors ``L``/``R`` in the cartoon)
    are ordered as follows: ``(uL/uR, vL/vR, dL/dR)`` which means "(up, virtual, down)".
    """

    def __init__(
        self,
        left_environment: np.ndarray,
        mpo_tensor_left: np.ndarray,
        mpo_tensor_right: np.ndarray,
        right_environment: np.ndarray,
    ):
        """Initialises an effective operator tensor network.

        Parameters
        ----------
        left_environment : np.ndarray
            The left environment for the effective operator.
        mpo_tensor_left : np.ndarray
            The left MPO tensor.
        mpo_tensor_right : np.ndarray
            The right MPO tensor.
        right_environment : np.ndarray
            The right environment for the effective operator.
        """

        if len(left_environment.shape) != 3:
            raise ValueError(
                "A valid left environment tensor must have 3 legs"
                f"while the one given has {len(left_environment.shape)}."
            )
        if len(mpo_tensor_left.shape) != 4:
            raise ValueError(
                "A valid mpo left tensor must have 4 legs"
                f"while the one given has {len(mpo_tensor_left.shape)}."
            )
        if len(mpo_tensor_right.shape) != 4:
            raise ValueError(
                "A valid mpo right tensor must have 4 legs"
                f"while the one given has {len(mpo_tensor_right.shape)}."
            )
        if len(right_environment.shape) != 3:
            raise ValueError(
                "A valid right environment tensor must have 3 legs"
                f"while the one given has {len(right_environment.shape)}."
            )

        self.left_environment = left_environment
        self.right_environment = right_environment
        self.mpo_tensor_left = mpo_tensor_left
        self.mpo_tensor_right = mpo_tensor_right
        chi_1, chi_2 = (
            left_environment.shape[2],
            right_environment.shape[2],
        )
        d_1, d_2 = (
            mpo_tensor_left.shape[3],
            mpo_tensor_right.shape[3],
        )
        self.x_shape = (chi_1, d_1, d_2, chi_2)
        self.shape = (chi_1 * d_1 * d_2 * chi_2, chi_1 * d_1 * d_2 * chi_2)
        self.dtype = mpo_tensor_left.dtype
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Performs matrix-vector multiplication.

        Computes ``effective_operator * |x> = |x'>``.
        This function is being used by :func:`scipy.sparse.linalg.eigsh` to diagonalise
        the effective operator with the Lanczos method, without generating the full matrix.

        Parameters
        ----------
        x : np.ndarray
            The two-site tensor on which acts an effective operator.
        """

        two_site_tensor = np.reshape(x, self.x_shape)

        if len(two_site_tensor.shape) != 4:
            raise ValueError(
                f"A valid two-site tensor must have 4 legs"
                f"while the one given has {len(two_site_tensor.shape)}."
            )

        einsum_string = "ijkl, mni, nopj, oqrk, sql -> mprs"
        two_site_tensor = contract(
            einsum_string,
            two_site_tensor,
            self.left_environment,
            self.mpo_tensor_left,
            self.mpo_tensor_right,
            self.right_environment,
            optimize=[(0, 1), (0, 3), (0, 2), (0, 1)],
        )

        return np.reshape(two_site_tensor, self.shape[0])


class DMRG:
    """Class storing the DMRG methods.

    Class holding the Density Matrix Renormalisation Group algorithm with two-site updates
    for a finite-size system with open-boundary conditions.

    Attributes
    ----------
    mps
        MPS serving as a current approximation of the target state.
    mpo
        The MPO of which the target state is to be computed.
        Each tensor in the MPO list has legs ``(vL, vR, pU, pD)``,
        where ``v`` stands for "virtual", ``p`` -- for "physical",
        and ``L``, ``R``, ``U``, ``D`` -- for "left", "right", "up", "down" accordingly.
    chi_max
        The highest bond dimension of an MPS allowed.
    mode
        Available options:
            | ``LM`` : Largest (in magnitude) eigenvalues.
            | ``SM`` : Smallest (in magnitude) eigenvalues.
            | ``LA`` : Largest (algebraic) eigenvalues.
            | ``SA`` : Smallest (algebraic) eigenvalues.
    cut :
        Truncation tolerance for tensor approximation
    silent :
        Whether to show/hide the progress bar.
    """

    def __init__(
        self,
        mps: Union[ExplicitMPS, CanonicalMPS],
        mpo: list[np.ndarray],
        chi_max: np.int16 = 1e4,
        cut: np.float64 = 1e-6,
        mode: str = "SA",
        silent: bool = False,
        copy: bool = True,
    ):

        if len(mps) != len(mpo):
            raise ValueError(
                f"The MPS has length {len(mps)},"
                f"the MPO has length {len(mpo)},"
                "but the lengths should be equal."
            )
        for i, tensor in enumerate(mpo):
            if len(tensor.shape) != 4:
                raise ValueError(
                    f"A valid MPO tensor must have 4 legs while tensor {i} has {len(tensor.shape)}."
                )
        if mode not in ["SA", "LA", "SM", "LM"]:
            raise ValueError("Invalid eigensolver mode given.")

        self.mps = mps
        if copy:
            self.mps = mps.copy()
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.right_canonical()
        self.left_environments = [None] * len(mps)
        self.right_environments = [None] * len(mps)
        self.mpo = mpo
        self.chi_max = chi_max
        self.cut = cut
        self.mode = mode
        self.silent = silent

        start_bond_dim = self.mpo[0].shape[0]
        chi = mps.tensors[0].shape[0]
        left_environment = np.zeros([chi, start_bond_dim, chi], dtype=np.float64)
        right_environment = np.zeros([chi, start_bond_dim, chi], dtype=np.float64)
        left_environment[:, 0, :] = np.eye(chi, dtype=np.float64)
        right_environment[:, start_bond_dim - 1, :] = np.eye(chi, dtype=np.float64)
        self.left_environments[0] = left_environment
        self.right_environments[-1] = right_environment
        for i in reversed(range(1, len(mps))):
            self.update_right_environment(i)

    def sweep(self):
        """
        One DMRG sweep.

        A method performing one DMRG sweep, which consists of
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

        effective_hamiltonian = EffectiveOperator(
            self.left_environments[i],
            self.mpo[i],
            self.mpo[j],
            self.right_environments[j],
        )

        if isinstance(self.mps, ExplicitMPS):
            initial_guess = self.mps.two_site_right_iso(i).reshape(
                effective_hamiltonian.shape[0]
            )
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.move_orth_centre(i)
            initial_guess = self.mps.two_site_tensor_next(i).reshape(
                effective_hamiltonian.shape[0]
            )

        _, eigenvectors = eigsh(
            effective_hamiltonian,
            k=1,
            which=self.mode,
            return_eigenvectors=True,
            v0=initial_guess,
            tol=1e-8,
        )
        x = eigenvectors[:, 0].reshape(effective_hamiltonian.x_shape)
        left_iso_i, singular_values_j, right_iso_j = split_two_site_tensor(
            x, chi_max=self.chi_max, cut=self.cut, renormalise=True
        )

        if isinstance(self.mps, CanonicalMPS):
            self.mps.tensors[i] = np.tensordot(
                left_iso_i, np.diag(singular_values_j), (2, 0)
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
            "ijk, lnjm, omp, knp -> ilo",
            right_iso,
            self.mpo[i],
            np.conjugate(right_iso),
            right_environment,
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
        self.right_environments[i - 1] = right_environment

    def update_left_environment(self, i: np.int16):
        """
        Compute the ``left_environment`` left of site ``i+1``
        from the ``left_environment`` left of site ``i``.
        """

        left_environment = self.left_environments[i]

        if isinstance(self.mps, ExplicitMPS):
            left_iso = self.mps.single_site_left_iso(i)
        if isinstance(self.mps, CanonicalMPS):
            self.mps = self.mps.move_orth_centre(i + 1)
            left_iso = self.mps.single_site_tensor(i)

        left_environment = contract(
            "ijk, lnjm, omp, ilo -> knp",
            left_iso,
            self.mpo[i],
            np.conjugate(left_iso),
            left_environment,
            optimize=[(0, 3), (0, 2), (0, 1)],
        )
        self.left_environments[i + 1] = left_environment

    def run(self, num_iter: np.int16 = 1):
        """
        Run the algorithm, i.e., run the ``sweep`` method for ``num_iter`` number of iterations.
        """

        for _ in tqdm(range(num_iter), disable=self.silent):
            self.sweep()
