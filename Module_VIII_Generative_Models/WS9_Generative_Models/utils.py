import numpy as np
import torch
import math
from torch.nn import functional as F


def get_target_indices(target, n_particles, dimensions):
    """
    Construct coordinate indices for a coupling layer.

    Given a tuple of target coordinate directions, return
    the flattened indices corresponding to

        - conditioned coordinates,
        - transformed coordinates.

    The coordinates are assumed to be stored in the order

        [x₁, y₁, z₁, x₂, y₂, z₂, ...]

    for a three-dimensional system (and similarly in
    other dimensions).

    Parameters
    ----------
    target : tuple[int]
        Coordinate directions to be transformed by the
        coupling layer.

        For example, in three dimensions

            (0,)     -> x coordinates
            (1,)     -> y coordinates
            (2,)     -> z coordinates
            (0, 2)   -> x and z coordinates

    n_particles : int
        Number of particles.

    dimensions : int
        Number of spatial dimensions.

    Returns
    -------
    conditioned_indices : np.ndarray
        Indices of the coordinates used as input to the
        conditioner network.

    transformed_indices : np.ndarray
        Indices of the coordinates transformed by the
        coupling layer.

    Examples
    --------
    For

        n_particles = 3
        dimensions = 2

    the flattened coordinate vector is

        [x₁, y₁, x₂, y₂, x₃, y₃].

    Using

        target = (0,)

    yields

        transformed_indices = [0, 2, 4]
        conditioned_indices = [1, 3, 5].
    """

    coordinate_indices =  np.arange(n_particles*dimensions)

    # Start from all coordinates conditioned.
    mask = np.ones(n_particles*dimensions, dtype=bool)
    
    # Mark the target coordinate directions
    # as transformed.
    for indx in target:
        mask[indx::dimensions] = 0

    return coordinate_indices[mask], coordinate_indices[~mask]


def searchsorted(bin_locations, inputs, eps=1e-6):

    bin_locations[..., -1] += eps

    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def torch_ns_cbrt(x):
    
    ans = torch.sign(x)*torch.exp(torch.log(torch.abs(x))/3.0)
    
    return ans


def torch_ns_sqrt(x):
    
    ans = torch.exp((torch.log(torch.abs(x))) / 2.0)
    
    return ans


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    enable_identity_init=False,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        print(torch.min(inputs), torch.max(inputs))
        raise Exception("Input Outside Domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    if enable_identity_init: #flow is the identity if initialized with parameters equal to zero
        beta = np.log(2) / (1 - min_derivative)
    else: #backward compatibility
        beta = 1
    derivatives = min_derivative + F.softplus(unnormalized_derivatives, beta=beta)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
    

def radial_distribution_function(
    configurations,
    system,
    cutoff=None,
    n_bins=100,
    batch_size=None,
    log_weights=None,
):
    """
    Compute the radial distribution function g(r).

    Parameters
    ----------
    configurations : torch.Tensor
        Configurations with shape

            [n_samples, system.dofs]

    system : LennardJones3D or EinsteinCrystal3D
        System providing

            n_particles
            dimensions
            box_length
            volume

    cutoff : float, optional
        Maximum distance included in the RDF.

        If None,

            cutoff = box_size / 2

    n_bins : int, default=100
        Number of histogram bins.

    batch_size : int, optional
        Number of configurations processed at once.

    log_weights : torch.Tensor, optional
        Log importance weights used for reweighting.

    Returns
    -------
    r : np.ndarray
        Bin centers.

    g_r : np.ndarray
        Radial distribution function.
    """

    if cutoff is None:
        cutoff = 0.5 * system.box_size

    n_samples = configurations.shape[0]

    if batch_size is None:
        batch_size = n_samples

    bin_edges = np.linspace(
        0.0,
        cutoff,
        n_bins + 1,
    )

    bin_width = bin_edges[1] - bin_edges[0]

    rdf_histogram = np.zeros(
        n_bins,
        dtype=np.float64,
    )

    for start in range(
        0,
        n_samples,
        batch_size,
    ):

        stop = min(
            start + batch_size,
            n_samples,
        )

        configuration_batch = (
            configurations[start:stop]
        )

        current_batch_size = (
            configuration_batch.shape[0]
        )

        # --------------------------------------------------
        # Pairwise distances
        # --------------------------------------------------

        positions = configuration_batch.view(
            -1,
            system.n_particles,
            system.dimensions,
        )

        pairwise_displacements = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        pairwise_displacements -= (
            system.box_length
            * torch.round(
                pairwise_displacements
                / system.box_length
            )
        )

        distances = torch.sqrt(
            torch.sum(
                pairwise_displacements**2,
                dim=-1,
            )
        )

        interaction_mask = (
            (distances > 0.0)
            &
            (distances < cutoff)
        )

        distances = distances[
            interaction_mask
        ]

        # --------------------------------------------------
        # Optional reweighting
        # --------------------------------------------------

        if log_weights is not None:

            batch_log_weights = (
                log_weights[start:stop]
            )

            weights = torch.exp(
                batch_log_weights
                - batch_log_weights.max()
            )

            pair_weights = (
                weights.view(
                    current_batch_size,
                    1,
                    1,
                )
                .expand(
                    current_batch_size,
                    system.n_particles,
                    system.n_particles,
                )
            )

            pair_weights /= weights.mean()

            pair_weights = pair_weights[
                interaction_mask
            ].cpu().numpy()

        else:

            pair_weights = None

        batch_histogram, _ = np.histogram(
            distances.cpu().numpy(),
            bins=bin_edges,
            weights=pair_weights,
        )

        rdf_histogram += batch_histogram

    # --------------------------------------------------
    # Shell volumes
    # --------------------------------------------------

    shell_volumes = (
        4.0
        * np.pi
        / 3.0
        * (
            bin_edges[1:]**3
            - bin_edges[:-1]**3
        )
    )

    density = (
        system.n_particles
        / system.volume
    )

    rdf = (
        rdf_histogram
        / n_samples
        / (
            shell_volumes
            * system.n_particles
            * density
        )
    )

    bin_centers = (
        bin_edges[:-1]
        + 0.5 * bin_width
    )

    return bin_centers, rdf