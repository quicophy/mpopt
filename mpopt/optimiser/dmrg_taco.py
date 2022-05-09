"""
This module contains the DMRG class. Inspired by TenPy.
"""

from copy import deepcopy

import numpy as np
import scipy.sparse
from opt_einsum import contract
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from mpopt.utils.utils import split_two_site_tensor, numpy_dense_to_pytaco_sparse
import pytaco as pt


class EffectiveHamiltonian(scipy.sparse.linalg.LinearOperator):
    """
    To fully use the advantage of :module:`scipy.sparse.linalg`, when we will be computing
    eigenvectors of local effective Hamiltonians, we will need a special class for them.

    To be diagonalised in `DMRG.update_bond`.

    .--uL                      uR--.
    |         i            j       |
    |    vL   |     b      |   vR  |
    (L)----(mpo[i])----(mpo[j])----(R)
    |         |            |       |
    |         i*           j*      |
    .--dL                      dR--.

    """

    def __init__(self, left_environment, mpo_1, mpo_2, right_environment):
        self.left_environment = left_environment
        self.right_environment = right_environment
        self.mpo_1 = mpo_1
        self.mpo_2 = mpo_2
        chi_1, chi_2 = (
            left_environment.shape[2],
            right_environment.shape[2],
        )
        d_1, d_2 = (
            mpo_1.shape[3],
            mpo_2.shape[3],
        )
        self.x_shape = (chi_1, d_1, d_2, chi_2)
        self.shape = (chi_1 * d_1 * d_2 * chi_2, chi_1 * d_1 * d_2 * chi_2)
        if mpo_1.dtype == pt.double: 
            self.dtype='f8'
        elif mpo_1.dtype == pt.float: 
            self.dtype='f4'
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, x):
        """
        Calculate |x'> = H_eff |x>.
        This function is used by :func:scipy.sparse.linalg.eigsh` to diagonalise
        the effective Hamiltonian with a Lanczos method, withouth generating the full matrix.
        """
        x_r = x.reshape(self.x_shape) #create reshaped array but shared the mem
        einsum_string = "mni, ijkl, nopj, oqrk, sql -> mprs" 
        res = pt.einsum(
            einsum_string,
            self.left_environment,
            x_r,
            self.mpo_1,
            self.mpo_2,
            self.right_environment,
            #optimize=[(0, 1), (0, 3), (0, 2), (0, 1)], #TODO: optimize the contraction path
        )
        
        return np.array(res).reshape(self.shape[0])


class DMRG:
    """
    Class holding the Density Matrix Renormalisation Group algorithm with two-site updates (DMRG-2)
    for a finite-size system with open-boundary conditions.

    Parameters:
        mps, model, chi_max, mode, tolerance:
            Same as attributes.

    Attributes:
        mps : MPS given as an instance of the ExplicitMPS class, which serves as
            a current approximation of the ground state.
        mpo : MPO, list of ndarrays[ndim=4]
            The MPO of which the groundstate is to be computed.
            Each tensor in the MPO list has legs (vL, vR, pU, pD), where v stands for "virtual",
            p -- for "physical", and L, R, U, D -- for "left", "right", "up", "down" accordingly.
        chi_max : int
            The highest bond dimension of an MPS allowed.
        mode : str, which mode of the eigensolver to use
            Available options:
                'LM' : Largest (in magnitude) eigenvalues.
                'SM' : Smallest (in magnitude) eigenvalues.
                'LA' : Largest (algebraic) eigenvalues.
                'SA' : Smallest (algebraic) eigenvalues.
        cut : float
            The lower boundary of the spectrum.
            All the singular values smaller than that will be discarded.
        left_environments, right_environments : lists of ndarrays[ndim=3]
            Left and right parts ("environments") of the effective Hamiltonian.
            Each left_environments[i] has legs (uL, vL, dL),
            right_environments[i] has legs (uR, vR, dR),
            where "u", "d", and "v" denote "up", "down", and "virtual" accordingly.
        silent : bool
            Whether to show/hide the progress bar.

            .--uL            uR--.
            |                    |
            |  vL   |    |   vR  |
            (L)----()----()----(R)
            |       |     |      |
            |                    |
            .--dL            dR--.
    """

    def __init__(self, mps, mpo, chi_max, cut, mode, silent=False, copy=True, threads=None):
        if threads is None:
            import os
            pt.set_num_threads(int(os.cpu_count()))
        else:
            pt.set_num_threads(threads)
        if not silent: 
            print(f"Initializing DMRG solver with {pt.get_num_threads()} threads.")
        if len(mps) != len(mpo):
            raise ValueError(
                f"The MPS has length ({len(mps)}), "
                f"the MPO has length ({len(mpo)}), "
                "but the lengths should be equal."
            )
        self.mps = mps
        if copy:
            self.mps = deepcopy(mps)
        self.left_environments = [None] * len(mps)
        self.right_environments = [None] * len(mps)
        self.mpo = [numpy_dense_to_pytaco_sparse(x) for x in mpo]
        self.chi_max = chi_max
        self.cut = cut
        self.mode = mode
        self.silent = silent

        # Initialise left and right environments.
        start_bond_dim = self.mpo[0].shape[0]
        chi = mps.tensors[0].shape[0]
        left_environment = pt.tensor([chi, start_bond_dim, chi], pt.compressed, dtype=pt.double)
        right_environment = pt.tensor([chi, start_bond_dim, chi], pt.compressed, dtype=pt.double)
        for i in range(chi):
            left_environment.insert([i,0,i], 1)
            right_environment.insert([i,start_bond_dim-1,i], 1)
        #left_environment = np.zeros([chi, start_bond_dim, chi], dtype=np.float64)
        #right_environment = np.zeros([chi, start_bond_dim, chi], dtype=np.float64)
        #left_environment[:, 0, :] = np.eye(chi, dtype=np.float64)
        #right_environment[:, start_bond_dim - 1, :] = np.eye(chi, dtype=np.float64)
        self.left_environments[0] = right_environment
        self.right_environments[-1] = right_environment

        # Update necessary right environments.
        for i in reversed(range(1, len(mps))):
            self.update_right_environment(i)

    def sweep(self):
        """
        A method performing one DMRG sweep, which consists of
        two series of `update_bond` sweeps which go back and forth.
        """

        # from left to right
        for i in range(self.mps.nsites - 1):
            self.update_bond(i)

        # from right to left
        for i in reversed(range(self.mps.nsites - 1)):
            self.update_bond(i)

    def update_bond(self, i):
        """
        A method which updates the bond between site `i` and `i+1`.
        """

        # get the effective Hamiltonian, which will be diagonalised during the update bond step:
        #
        #    .--uL                      uR--.
        #    |         i            j       |
        #    |    vL   |     b      |   vR  |
        #    (L)----(mpo[i])----(mpo[j])----(R)
        #    |         |            |       |
        #    |         i*           j*      |
        #    .--dL                      dR--.
        #
        # left_environment: uL, vL, dL
        # right_environment: uR, vR, dR
        # mpo[i]: vL, b, i, i*
        # mpo[j]: b, vR, j, j*

        j = i + 1

        effective_hamiltonian = EffectiveHamiltonian(
            self.left_environments[i],
            self.mpo[i],
            self.mpo[j],
            self.right_environments[j],
        )

        # Diagonalise the effective Hamiltonian, find its ground state.
        initial_guess = self.mps.two_site_right_tensor(i).reshape(
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

        # Put back into MPS
        self.mps.tensors[i] = np.tensordot(
            np.linalg.inv(np.diag(self.mps.singular_values[i])), left_iso_i, (1, 0)
        )
        self.mps.tensors[j] = np.tensordot(
            right_iso_j, np.linalg.inv(np.diag(self.mps.singular_values[j + 1])), (2, 0)
        )
        self.mps.singular_values[j] = singular_values_j
        self.update_left_environment(i)
        self.update_right_environment(j)

    def update_right_environment(self, i):
        """
        Compute `right_environment` right of site `i-1` from `right_environment` right of site `i`.
        """

        right_environment = self.right_environments[i]
        right_iso = pt.from_array(self.mps.single_site_right_iso(i))
        right_environment = pt.einsum(
            "ijk, lnjm, omp, knp -> ilo",
            right_iso,
            self.mpo[i],
            right_iso, #np.conj(right_iso), TODO: no complex...
            right_environment,
            #optimize=[(0, 3), (0, 2), (0, 1)], TODO: optimize the contraction path
        )
        right_environment.evaluate()
        self.right_environments[i - 1] = right_environment

    def update_left_environment(self, i):
        """
        Compute `left_environment` left of site `i+1` from `left_environment` left of site `i`.
        """

        left_environment = self.left_environments[i]
        left_iso = pt.from_array(self.mps.single_site_left_iso(i))
        left_iso_conj = pt.from_array(np.conj(self.mps.single_site_left_iso(i)), copy=True) #TODO: PyTaco can't work with complex...
        left_environment = pt.einsum(
            "ijk, lnjm, omp, ilo -> knp",
            left_iso,
            self.mpo[i],
            left_iso_conj,
            left_environment,
            #optimize=[(0, 3), (0, 2), (0, 1)], TODO: optimize the contraction path
        )
        left_environment.evaluate()
        self.left_environments[i + 1] = left_environment

    def run(self, num_iter):
        """
        Run the algorithm, i.e., run the `sweep` method for `num_iter` number of times.
        """

        for _ in tqdm(range(num_iter), disable=self.silent):
            self.sweep()
