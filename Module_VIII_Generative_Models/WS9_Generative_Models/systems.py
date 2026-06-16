import numpy as np
import torch


class EinsteinCrystal3D:
    """
    Three-dimensional Einstein crystal on an FCC lattice.

    Each particle fluctuates independently around its FCC lattice
    site according to a truncated isotropic Gaussian distribution.

    The harmonic energy is

        U(x) = Σ_i |x_i - r_i^(0)|² / (2 σ²)

    where r_i^(0) denotes the reference FCC lattice position.

    The Gaussian is truncated at a fixed radius around each lattice
    site to prevent particle permutations.

    Parameters
    ----------
    n_particles : int
        Number of particles. Must satisfy

            n_particles = 4 * n_unit_cells³

        for an integer number of FCC unit cells.

    rho : float
        Number density.

    displacement_std : float
        Standard deviation of the Gaussian fluctuations.

    truncation_radius : float
        Maximum allowed displacement from a lattice site.

    device : torch.device
        Device used for tensor computations.

    tol : float, default=1e-6
        Numerical tolerance.
    """

    def __init__(
        self,
        n_particles,
        rho,
        displacement_std,
        truncation_radius,
        device,
        tol=1e-6,
    ):

        # --------------------------------------------------
        # Basic system information
        # --------------------------------------------------

        self.n_particles = n_particles
        self.dimensions = 3
        self.dofs = 3 * n_particles

        self.device = device
        self.PBC = True

        self.rho = rho

        self.displacement_std = displacement_std
        self.truncation_radius = truncation_radius

        self.tol = tol
        self.tol_sq = tol**2

        # --------------------------------------------------
        # FCC geometry
        # --------------------------------------------------

        self.n_unit_cells = round(
            (n_particles / 4) ** (1 / 3)
        )

        assert (
            4 * self.n_unit_cells**3
            == n_particles
        ), (
            "For an FCC crystal the number of particles "
            "must satisfy N = 4 * n_unit_cells³."
        )

        self.lattice_constant = (
            4.0 / rho
        ) ** (1.0 / 3.0)

        self.box_size = (
            self.n_unit_cells
            * self.lattice_constant
        )

        self.box_length = torch.full(
            (3,),
            self.box_size,
            dtype=torch.float32,
            device=device,
        )

        self.volume = self.box_size**3

        density = (
            self.n_particles
            / self.volume
        )

        assert (
            abs(density - self.rho)
            < self.tol
        )

        # --------------------------------------------------
        # Reference FCC configuration
        # --------------------------------------------------

        self.reference_configuration = (
            self.init_conf()
            .view(
                self.n_particles,
                self.dimensions,
            )
        )

        self.x0 = None

    def energy(self, x):
        """
        Compute the Einstein crystal energy.

        Parameters
        ----------
        x : torch.Tensor
            Configurations with shape

                [batch_size, dofs]

        Returns
        -------
        torch.Tensor
            Harmonic energies with shape

                [batch_size]
        """

        positions = x.view(
            -1,
            self.n_particles,
            self.dimensions,
        )

        displacements = (
            positions
            - self.reference_configuration
        )

        displacements -= (
            self.box_size
            * torch.round(
                displacements
                / self.box_size
            )
        )

        squared_displacements = torch.sum(
            displacements**2,
            dim=-1,
        )

        total_energy = (
            0.5
            / self.displacement_std**2
            * torch.sum(
                squared_displacements,
                dim=1,
                keepdim=True,
            )
        )

        return total_energy
    
    def init_conf(self, as_numpy=False):
        """
        Generate a perfect FCC crystal configuration.

        Returns
        -------
        np.ndarray or torch.Tensor
            FCC lattice positions.
        """

        fcc_basis = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )

        positions = []

        for i in range(self.n_unit_cells):
            for j in range(self.n_unit_cells):
                for k in range(self.n_unit_cells):

                    cell_origin = (
                        self.lattice_constant
                        * np.array(
                            [i, j, k],
                            dtype=np.float32,
                        )
                    )

                    for basis_atom in fcc_basis:

                        positions.append(
                            cell_origin
                            + self.lattice_constant
                            * basis_atom
                        )

        positions = np.asarray(
            positions,
            dtype=np.float32,
        )

        positions -= (
            0.5 * self.box_size
        )

        if as_numpy:
            return positions

        return torch.tensor(
            positions.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )

    def _sample_space(
        self,
        n_walkers,
        beta=None,
        print_acceptance=False,
    ):
        """
        Sample configurations from the truncated Gaussian prior.

        Parameters
        ----------
        n_walkers : int
            Number of configurations to generate.

        beta : ignored
            Included only for API compatibility with
            MetropolisMonteCarlo.

        Returns
        -------
        dict
            Dictionary containing sampled configurations
            and energies.
        """

        reference_positions = (
            self.reference_configuration
            .unsqueeze(0)
            .repeat(
                n_walkers,
                1,
                1,
            )
        )

        displacements = torch.zeros_like(
            reference_positions
        )

        accepted = torch.zeros(
            (
                n_walkers,
                self.n_particles,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        total_proposals = 0
        total_accepts = 0

        while not torch.all(accepted):

            proposals = (
                self.displacement_std
                * torch.randn(
                    n_walkers,
                    self.n_particles,
                    self.dimensions,
                    device=self.device,
                )
            )

            radii = torch.linalg.norm(
                proposals,
                dim=-1,
            )

            new_accepts = (
                radii
                < self.truncation_radius
            ) & (~accepted)

            total_proposals += (
                (~accepted).sum().item()
            )

            total_accepts += (
                new_accepts.sum().item()
            )

            displacements[
                new_accepts
            ] = proposals[
                new_accepts
            ]

            accepted |= new_accepts

        positions = (
            reference_positions
            + displacements
        )

        configurations = positions.reshape(
            n_walkers,
            self.dofs,
        )

        energies = self.energy(
            configurations
        )

        acceptance = (
            total_accepts
            / total_proposals
        )

        if print_acceptance:
            print(
                f"Prior acceptance = "
                f"{acceptance:.3f}"
            )

        return {
            "x": configurations,
            "energy": energies,
            "acceptance": acceptance,
        }

    def sample(
            self, 
            n_samples,
            beta=None,
            sampler=None,
            return_energy=False,
            print_acceptance=False,
            ):

        if sampler is None:
            results = self._sample_space(
                n_walkers=n_samples,
                beta=beta,
                print_acceptance=print_acceptance,
            )
        else:
            results = sampler._sample_space(
                n_walkers=n_samples,
                beta=beta,
            )

        if return_energy:
            return results["x"], results["energy"]
        
        return results["x"]
    


class LennardJones3D:
    """
    Three-dimensional Lennard-Jones crystal in reduced units.

    This class represents a Lennard-Jones solid with periodic boundary
    conditions. All quantities are expressed in standard Lennard-Jones
    reduced units,

        σ = 1
        ε = 1
        k_B = 1

    so that all distances, energies, temperatures, and densities are
    dimensionless.

    The system is initialized as a face-centered cubic (FCC) crystal.
    Since an FCC unit cell contains four particles, the number of
    particles must satisfy

        N = 4 n_cells³

    for some integer number of unit cells n_cells.

    The lattice constant is determined from the requested density,

        ρ = 4 / a³,

    and the simulation box is chosen to be cubic with side length

        L = n_cells a.

    Parameters
    ----------
    n_particles : int
        Number of particles in the system. Must satisfy

            n_particles = 4 * n_cells³

        for an integer number of FCC unit cells.

    rho : float
        Number density in Lennard-Jones reduced units.

    device : torch.device
        Device used for tensor computations.

    cutoff : float, optional
        Lennard-Jones cutoff radius.

        If None, the cutoff is chosen as

            min(2.5, 0.49 * box_size)

        which guarantees compatibility with the minimum-image
        convention.

    linearization_radius : float, optional
        Distance below which the Lennard-Jones potential is replaced
        by its tangent line. This prevents numerical instabilities
        caused by extremely close particles and is mainly useful when
        training differentiable models.

    long_range_corrections : bool, default=True
        Whether to include the standard long-range correction to the
        potential energy.

    tol : float, default=1e-6
        Numerical tolerance used for consistency checks.

    Attributes
    ----------
    n_particles : int
        Number of particles.

    dimensions : int
        Spatial dimension (always 3).

    dofs : int
        Total number of Cartesian degrees of freedom,
        dofs = 3 * n_particles.

    rho : float
        Number density.

    lattice_constant : float
        FCC lattice constant.
    
    n_unit_cells : int
        Number of FCC unit cells along each edge of the cubic
        simulation box.

    box_size : float
        Side length of the cubic simulation box.

    box_length : torch.Tensor, shape (3,)
        Box lengths along the three Cartesian directions.

    volume : float
        Simulation box volume.

    cutoff : float
        Interaction cutoff radius.

    cutoff_squared : float
        Squared cutoff radius.

    cutoff_energy_shift : float
        Constant energy shift applied so that the Lennard-Jones
        potential is continuous at the cutoff.

    energy_tail_correction : float
        Long-range energy correction per particle.

    linearization_radius : float or None
        Radius below which the potential is linearized.

    linearization_slope : float
        Slope of the linearized potential.

    linearization_energy : float
        Energy offset used in the linearized potential.

    PBC : bool
        Whether periodic boundary conditions are enabled.
    """

    def __init__(
        self,
        n_particles,
        rho,
        device,
        cutoff=None,
        linearization_radius=None,
        long_range_corrections=True,
        tol=1e-6,
    ):

        # --------------------------------------------------
        # Basic system information
        # --------------------------------------------------

        self.n_particles = n_particles
        self.dimensions = 3
        self.dofs = 3 * n_particles

        self.device = device
        self.PBC = True

        self.rho = rho

        # --------------------------------------------------
        # FCC simulation box
        # --------------------------------------------------
        #
        # For an FCC crystal:
        #
        #     rho = 4 / a^3
        #
        # where a is the lattice constant.
        #
        # We require the number of particles to correspond
        # exactly to an integer number of FCC unit cells:
        #
        #     N = 4 * n_cells^3
        #
        # --------------------------------------------------

        self.n_unit_cells = round((n_particles / 4) ** (1 / 3))

        assert (
            4 * self.n_unit_cells**3 == n_particles
        ), (
            "For an FCC crystal the number of particles "
            "must satisfy N = 4 * n_unit_cells^3."
        )

        self.lattice_constant = (
            4.0 / self.rho
        ) ** (1.0 / 3.0)

        self.box_size = (
            self.n_unit_cells * self.lattice_constant
        )

        self.box_length = torch.full(
            (3,),
            self.box_size,
            dtype=torch.float32,
            device=device,
        )

        self.volume = self.box_size**3

        density = self.n_particles / self.volume

        assert (
            abs(density - self.rho) < tol
        ), (
            f"Density mismatch: "
            f"rho = {self.rho}, N/V = {density}"
        )

        # --------------------------------------------------
        # Lennard-Jones cutoff
        # --------------------------------------------------
        #
        # The cutoff must satisfy
        #
        #     r_cut < L/2
        #
        # in order for the minimum-image convention to be
        # valid.
        #
        # --------------------------------------------------

        self.long_range_corrections = long_range_corrections

        if cutoff is None:

            self.cutoff = min(
                2.5,
                0.49 * self.box_size,
            )

        else:

            self.cutoff = cutoff

        assert (
            self.cutoff < 0.5 * self.box_size
        ), (
            "Cutoff must be smaller than half "
            "the box length."
        )

        self.cutoff_squared = self.cutoff**2

        # Constant energy shift used to make the
        # potential continuous at the cutoff.

        self.cutoff_energy_shift = 4.0 * (
            self.cutoff**(-12)
            - self.cutoff**(-6)
        )

        # --------------------------------------------------
        # Long-range energy correction
        # --------------------------------------------------
        #
        # Energy correction per particle obtained from the
        # standard homogeneous-fluid approximation.
        #
        # Total correction:
        #
        #     U_tail = N * energy_tail_correction
        #
        # --------------------------------------------------

        if self.long_range_corrections:

            self.energy_tail_correction = (
                8.0
                * np.pi
                * self.rho
                / 9.0
                * (
                    self.cutoff**(-9)
                    - 3.0 * self.cutoff**(-3)
                )
            )

        else:

            self.energy_tail_correction = 0.0

        # --------------------------------------------------
        # Short-distance linearization
        # --------------------------------------------------
        #
        # For very small separations the Lennard-Jones
        # potential becomes extremely large and can cause
        # unstable gradients.
        #
        # Below linearization_radius the potential is
        # replaced by its tangent line.
        #
        # --------------------------------------------------

        self.linearization_radius = linearization_radius

        if self.linearization_radius is not None:

            assert (
                self.linearization_radius < self.cutoff
            ), (
                "linearization_radius must be smaller "
                "than the cutoff."
            )

            r = self.linearization_radius

            self.linearization_slope = (
                -24.0
                / r
                * (
                    2.0 * r**(-12)
                    - r**(-6)
                )
            )

            self.linearization_energy = (
                4.0
                * (
                    r**(-12)
                    - r**(-6)
                )
                - self.cutoff_energy_shift
            )

        # --------------------------------------------------
        # Numerical parameters
        # --------------------------------------------------

        self.tol = tol
        self.tol_sq = tol**2

        # Cached configuration used by samplers

        self.x0 = None

    def energy(self, x):
        """
        Compute the Lennard-Jones potential energy.

        Parameters
        ----------
        x : torch.Tensor
            Particle coordinates with shape

                [batch_size, n_particles * 3]

        Returns
        -------
        torch.Tensor
            Potential energy of each configuration with shape

                [batch_size]
        """

        # --------------------------------------------------
        # Reshape coordinates
        # --------------------------------------------------

        positions = x.view(
            -1,
            self.n_particles,
            self.dimensions,
        )

        # --------------------------------------------------
        # Pairwise displacement vectors
        # --------------------------------------------------

        pairwise_displacements = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        # Minimum-image convention

        pairwise_displacements -= (
            self.box_length
            * torch.round(
                pairwise_displacements
                / self.box_length
            )
        )

        # --------------------------------------------------
        # Pairwise distances
        # --------------------------------------------------

        squared_distances = torch.clamp(
            torch.sum(
                pairwise_displacements**2,
                dim=-1,
            ),
            min=self.tol_sq,
        ).unsqueeze(-1)

        inverse_r6 = squared_distances**(-3)
        inverse_r12 = inverse_r6**2

        # --------------------------------------------------
        # Shifted Lennard-Jones potential
        # --------------------------------------------------

        pairwise_energy = (
            4.0
            * (
                inverse_r12
                - inverse_r6
            )
            - self.cutoff_energy_shift
        )

        # --------------------------------------------------
        # Interaction mask
        # --------------------------------------------------

        interaction_mask = (
            (squared_distances < self.cutoff_squared)
            &
            (squared_distances > self.tol_sq)
        )

        pairwise_energy = torch.where(
            interaction_mask,
            pairwise_energy,
            torch.zeros_like(pairwise_energy),
        )

        # --------------------------------------------------
        # Optional short-range linearization
        # --------------------------------------------------

        if self.linearization_radius is not None:

            distances = torch.sqrt(
                squared_distances
            )

            linearized_energy = (
                self.linearization_slope
                * (
                    distances
                    - self.linearization_radius
                )
                + self.linearization_energy
            )

            linearization_mask = (
                (distances < self.linearization_radius)
                &
                (distances > self.tol)
            )

            pairwise_energy = torch.where(
                linearization_mask,
                linearized_energy,
                pairwise_energy,
            )

        # --------------------------------------------------
        # Total energy
        # --------------------------------------------------

        total_energy = (
            0.5
            * torch.sum(
                pairwise_energy,
                dim=(1, 2),
            )
            + self.n_particles
            * self.energy_tail_correction
        )

        return total_energy
    
    def init_conf(self, as_numpy=False):
        """
        Generate a perfect FCC crystal configuration.

        The configuration is constructed by placing the four FCC basis
        atoms in each cubic unit cell and repeating the unit cell
        throughout the simulation box.

        Coordinates are centered in the simulation box so that particles
        lie approximately in the interval

            [-L/2, L/2]

        along each Cartesian direction.

        Parameters
        ----------
        as_numpy : bool, default=False
            If True, return an array of shape

                [n_particles, 3].

            Otherwise return a flattened PyTorch tensor of shape

                [3 * n_particles].

        Returns
        -------
        np.ndarray or torch.Tensor
            Initial FCC configuration.
        """

        # FCC basis positions inside a unit cell
        fcc_basis = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )

        positions = []

        for i in range(self.n_unit_cells):
            for j in range(self.n_unit_cells):
                for k in range(self.n_unit_cells):

                    cell_origin = (
                        self.lattice_constant
                        * np.array([i, j, k], dtype=np.float32)
                    )

                    for basis_atom in fcc_basis:

                        positions.append(
                            cell_origin
                            + self.lattice_constant * basis_atom
                        )

        positions = np.asarray(
            positions,
            dtype=np.float32,
        )

        assert (
            positions.shape[0] == self.n_particles
        ), "Incorrect number of FCC lattice sites generated."

        # Center crystal in the simulation box
        positions -= 0.5 * self.box_size

        if as_numpy:
            return positions

        return torch.tensor(
            positions.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
    
    def sample(
        self,
        n_samples,
        beta,
        sampler,
        return_energy=False,
    ):
        
        results = sampler.sample_space(
            n_walkers=n_samples,
            beta=beta,
        )

        if return_energy:
            return results["x"], results["energy"]

        return results["x"] 