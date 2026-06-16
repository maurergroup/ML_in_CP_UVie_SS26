import numpy as np
import torch


class MetropolisMonteCarlo:
    """
    Metropolis Monte Carlo sampler for canonical (NVT) simulations.

    This class implements parallel single-particle Metropolis updates for a
    collection of independent walkers. Each walker samples configurations
    distributed according to the Boltzmann distribution

        p(x) ∝ exp(-β U(x))

    where U(x) is the potential energy of the system and β = 1/(k_B T).

    A Monte Carlo cycle consists of:

    1. Selecting one particle in each walker.
    2. Proposing a random displacement of that particle.
    3. Evaluating the energy difference ΔU.
    4. Accepting the move with probability

           p_acc = min(1, exp(-β ΔU))

    5. Applying periodic boundary conditions (if enabled).

    Parameters
    ----------
    system : object
        Physical system providing:

        - energy(x) -> tensor of shape [batch]
        - init_conf() -> tensor of shape [dofs]

        and the attributes:

        - n_particles
        - dimensions
        - dofs
        - device
        - PBC
        - box_length

    step_size : float
        Maximum displacement applied independently to each Cartesian
        coordinate of the selected particle.

    n_cycles : int
        Number of Monte Carlo cycles performed between returned samples.

    n_equilibration : int, optional
        Number of equilibration (burn-in) cycles. If None, defaults to
        10 * n_cycles.
    """

    def __init__(
        self,
        system,
        step_size,
        n_cycles,
        n_equilibration=None,
    ):

        assert step_size > 0
        assert n_cycles > 0

        self.system = system

        self.dimensions = system.dimensions
        self.n_particles = system.n_particles
        self.dofs = system.dofs
        self.device = system.device

        self.step_size = step_size
        self.n_cycles = n_cycles

        self.x0 = None

        self.equilibrated = False
        self.beta_last = None

        if n_equilibration is None:
            self.n_equilibration = 10 * n_cycles
        else:
            self.n_equilibration = n_equilibration

    def metropolis_cycle(self, x, u_x, beta, dx):
        """
        Perform a single Metropolis cycle.

        One particle is selected independently in each walker and displaced
        uniformly in the interval [-dx, dx] for each Cartesian coordinate.

        Parameters
        ----------
        x : torch.Tensor
            Current configurations of shape [n_walkers, dofs].

        u_x : torch.Tensor
            Current energies of shape [n_walkers].

        beta : float
            Inverse temperature.

        dx : float
            Proposal displacement magnitude.

        Returns
        -------
        x : torch.Tensor
            Updated configurations.

        u_x : torch.Tensor
            Updated energies.

        acceptance : float
            Fraction of accepted moves.
        """

        n_walkers = x.shape[0]

        # ------------------------------------------------------------
        # Generate proposal
        # ------------------------------------------------------------

        shift = torch.zeros(
            (n_walkers, self.dofs),
            device=self.device,
        )

        selected_particles = torch.randint(
            self.n_particles,
            size=(n_walkers,),
            device=self.device,
        )

        selected_dofs = torch.stack(
            [
                selected_particles * self.dimensions + i
                for i in range(self.dimensions)
            ],
            dim=0,
        )

        batch_idx = torch.arange(
            n_walkers,
            device=self.device,
        )

        random_displacement = (
            2.0 * torch.rand(
                (n_walkers, self.dimensions),
                device=self.device,
            )
            - 1.0
        ) * dx

        shift[
            batch_idx[:, None],
            selected_dofs.T,
        ] = random_displacement

        xp = x + shift

        # ------------------------------------------------------------
        # Evaluate proposed energies
        # ------------------------------------------------------------

        u_xp = self.system.energy(xp).squeeze()

        # ------------------------------------------------------------
        # Metropolis acceptance test
        # ------------------------------------------------------------

        delta_u = u_xp - u_x

        log_random = torch.log(
            torch.rand(
                n_walkers,
                device=self.device,
            )
        )

        accepted = log_random < (-beta * delta_u)

        # ------------------------------------------------------------
        # Apply periodic boundary conditions
        # ------------------------------------------------------------

        if accepted.any():

            xp_acc = xp[accepted].view(
                -1,
                self.n_particles,
                self.dimensions,
            )

            if self.system.PBC:

                xp_acc -= (
                    self.system.box_length
                    * torch.round(
                        xp_acc / self.system.box_length
                    )
                )

            x[accepted] = xp_acc.view(-1, self.dofs)
            u_x[accepted] = u_xp[accepted]

        acceptance = accepted.float().mean().item()

        return x, u_x, acceptance

    def sample_space(self, n_walkers, beta):
        """
        Generate approximately independent equilibrium configurations.

        If the sampler has already been used, the final configurations from
        the previous run are used as initial conditions. Otherwise the
        system initialization routine is called.

        A new equilibration stage is automatically performed whenever the
        temperature changes.

        Parameters
        ----------
        n_walkers : int
            Number of walkers (parallel Markov chains).

        beta : float
            Inverse temperature.

        Returns
        -------
        results : dict

            results["x"]
                Configurations of shape [n_walkers, dofs]

            results["energy"]
                Energies of shape [n_walkers]

            results["acceptance"]
                Acceptance ratio of the final cycle
        """

        # ------------------------------------------------------------
        # Re-equilibrate if temperature changes
        # ------------------------------------------------------------

        if self.beta_last != beta:
            self.equilibrated = False
            self.beta_last = beta

        # ------------------------------------------------------------
        # Initial configurations
        # ------------------------------------------------------------

        if self.x0 is None:

            x = torch.stack(
                [
                    self.system.init_conf()
                    for _ in range(n_walkers)
                ]
            )

        else:

            indices = np.random.choice(
                np.arange(self.x0.shape[0]),
                replace=(n_walkers > self.x0.shape[0]),
                size=n_walkers,
            )

            x = self.x0[indices]

        u_x = self.system.energy(x).squeeze()

        # ------------------------------------------------------------
        # Equilibration
        # ------------------------------------------------------------

        if not self.equilibrated:

            print(
                f"Equilibrating at T = {1.0 / beta:.4f}"
            )

            for _ in range(self.n_equilibration):

                x, u_x, acc = self.metropolis_cycle(
                    x,
                    u_x,
                    beta,
                    self.step_size,
                )

            self.equilibrated = True

            print(
                f"Equilibration acceptance = {acc:.3f}"
            )

        # ------------------------------------------------------------
        # Production sampling
        # ------------------------------------------------------------

        for _ in range(self.n_cycles):

            x, u_x, acc = self.metropolis_cycle(
                x,
                u_x,
                beta,
                self.step_size,
            )

        print(
            f"Production acceptance = {acc:.3f}"
        )

        self.x0 = x.clone()

        return {
            "x": x,
            "energy": u_x,
            "acceptance": acc,
        }