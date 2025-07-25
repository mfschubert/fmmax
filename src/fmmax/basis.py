# FMMAX
# Copyright (C) 2025 Martin F. Schubert

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Functions related to vectors and field expansion in the FMM scheme.

Copyright (c) Martin F. Schubert
"""

import dataclasses
import enum
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from fmmax import utils


def brillouin_zone_in_plane_wavevector(
    brillouin_grid_shape: Tuple[int, int],
    primitive_lattice_vectors: "LatticeVectors",
) -> jnp.ndarray:
    """Compute in-plane wavevectors suitable for Brillouin zone integration.

    The wavevectors are evenly spaced within the first Brillouin zone; for odd grid
    shapes, they subdivide the first Brillouin zone evenly. For even grid shapes, they
    are offset so ``(0, 0)`` is included among the wavevectors.

    Args:
        brillouin_grid_shape: The shape of the wavevector grid.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.

    Returns:
        The in-plane wavevectors, with shape ``brillouin_grid_shape + (2,)``.
    """
    if len(brillouin_grid_shape) != 2 or brillouin_grid_shape < (1, 1):
        raise ValueError(
            f"`brillouin_grid_shape` must be length-2 with positive values, "
            f"but got {brillouin_grid_shape}."
        )

    udim, vdim = brillouin_grid_shape
    i, j = jnp.meshgrid(
        jnp.arange(-(udim // 2), udim - (udim // 2)) / udim,
        jnp.arange(-(vdim // 2), vdim - (vdim // 2)) / vdim,
        indexing="ij",
    )
    assert i.shape == brillouin_grid_shape
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    ku = reciprocal_vectors.u
    kv = reciprocal_vectors.v
    return jnp.stack(
        [
            2 * jnp.pi * (i * ku[..., 0] + j * kv[..., 0]),
            2 * jnp.pi * (i * ku[..., 1] + j * kv[..., 1]),
        ],
        axis=-1,
    )


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Following code is Copyright (c) Meta Platforms, Inc. and affiliates.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Officially defines the x- and y- directions. By convention, the x-axis preceeds
# the y-axis in our array indexing scheme.
X: jnp.ndarray = jnp.array([1.0, 0.0], dtype=jnp.float32)
Y: jnp.ndarray = jnp.array([0.0, 1.0], dtype=jnp.float32)


@dataclasses.dataclass
class LatticeVectors:
    """Stores a pair of lattice vectors.

    Note that this is just a pair of 2-dimensional vectors, which may be either for
    the real-space lattice or the reciprocal space lattice, depending on usage.

    Attributes:
        u: The first primitive lattice vector.
        v: The second primitive lattice vector, with identical shape.
    """

    u: jnp.ndarray
    v: jnp.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.u, jnp.ndarray):
            if self.u.shape[-1] != 2 or self.v.shape[-1] != 2:
                raise ValueError(
                    f"`u` and `v` must have a trailing length of 2, but got shapes "
                    f"{self.u.shape} and {self.v.shape}."
                )

    @property
    def reciprocal(self) -> "LatticeVectors":
        """Returns the corresponding vectors on the reciprocal lattice."""
        return _reciprocal(self)


@dataclasses.dataclass
class Expansion:
    """Stores the expansion.

    The expansion consists of the integer coefficients of the reciprocal lattice vectors
    used in the Fourier expansion of fields, permittivities, etc. in the FMM scheme.

    Attributes:
        basis_coefficients: The integer coefficients of the primitive reciprocal lattice
            vectors, which generate the full set of reciprocal-space vectors in the
            expansion.
        num_terms: The number of terms in the expansion.
    """

    basis_coefficients: onp.ndarray[Any, Any]

    def __post_init__(self) -> None:
        if self.basis_coefficients.ndim != 2 or self.basis_coefficients.shape[-1] != 2:
            raise ValueError(
                f"`basis_coefficients` must have shape `(num, 2)` but got "
                f"{self.basis_coefficients.shape}."
            )

    def __hash__(self) -> int:
        return hash(self.basis_coefficients.tobytes())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Expansion):
            return False
        if self.basis_coefficients.shape != other.basis_coefficients.shape:
            return False
        return bool(onp.all(self.basis_coefficients == other.basis_coefficients))

    @property
    def num_terms(self) -> int:
        return self.basis_coefficients.shape[-2]


@enum.unique
class Truncation(enum.Enum):
    """Enumerates truncation modes."""

    #: Fourier orders are truncated based on the total magnitude of their
    #: associated wavevectors.
    CIRCULAR = "circular"

    #: Fourier orders are truncated based on the magnitude of their component along
    #: ``u`` and ``v`` directions, without regard for their total magnitude.
    PARALLELOGRAMIC = "parallelogramic"


def min_array_shape_for_expansion(expansion: Expansion) -> Tuple[int, int]:
    """Returns the minimum allowed shape compatible with `expansion`."""
    with jax.ensure_compile_time_eval():
        return (
            int(2 * onp.amax(onp.abs(expansion.basis_coefficients[:, 0])) + 1),
            int(2 * onp.amax(onp.abs(expansion.basis_coefficients[:, 1])) + 1),
        )


def validate_shape_for_expansion(shape: Tuple[int, ...], expansion: Expansion) -> None:
    """Validates that the shape is sufficient for the provided expansion."""
    min_shape = min_array_shape_for_expansion(expansion)
    if any([d < dmin for d, dmin in zip(shape[-2:], min_shape)]):
        raise ValueError(
            f"`shape` is insufficient for `expansion`, the minimum shape for the "
            f"final two axes is {min_shape} but got shape {shape}."
        )


def generate_expansion(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
    truncation: Truncation = Truncation.CIRCULAR,
) -> Expansion:
    """Generates the expansion for the specified real-space basis.

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a symmetric expansion, the total number of terms may differ from
            this value.
        truncation: The truncation to be used for the expansion. The default is
            ``Truncation.CIRCULAR``.

    Returns:
        The ``Expansion``. The basis coefficients of the expansion are sorted so that
        the zeroth-order term is first.
    """
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    if truncation == Truncation.CIRCULAR:
        basis_coefficients = _basis_coefficients_circular(
            reciprocal_vectors, approximate_num_terms
        )
    elif truncation == Truncation.PARALLELOGRAMIC:
        basis_coefficients = _basis_coefficients_parallelogramic(
            reciprocal_vectors, approximate_num_terms
        )
    else:
        raise ValueError(f"Unknown `truncation`, got {truncation}.")
    return Expansion(basis_coefficients)


def _reciprocal(lattice_vectors: LatticeVectors) -> LatticeVectors:
    """Computes the reciprocal vectors for the `basis`."""
    cross_product = _cross_product(lattice_vectors.u, lattice_vectors.v)
    uprime = (
        jnp.stack([lattice_vectors.v[..., 1], -lattice_vectors.v[..., 0]], axis=-1)
        / cross_product[..., jnp.newaxis]
    )
    vprime = (
        jnp.stack([-lattice_vectors.u[..., 1], lattice_vectors.u[..., 0]], axis=-1)
        / cross_product[..., jnp.newaxis]
    )
    assert uprime.shape == lattice_vectors.u.shape
    assert vprime.shape == lattice_vectors.v.shape
    return LatticeVectors(u=uprime, v=vprime)


def unit_cell_coordinates(
    primitive_lattice_vectors: LatticeVectors,
    shape: Tuple[int, int],
    num_unit_cells: Tuple[int, int] = (1, 1),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute spatial coordinates given the grid shape and number of unit cells.

    Args:
        primitive_lattice_vectors: The lattice vectors defining te unit cell.
        shape: The shape of the coordinates grid.
        num_unit_cells: Determines the number of unit cells for which to compute the
            coordinates. Default is ``(1, 1)``, corresponding to a single unit cell.

    Returns:
        The unit cell coordinates, with shape ``(shape[0] * num_unit_cells[0],
        shape[1] * num_unit_cells[1])``.
    """
    i_stop = num_unit_cells[0] * shape[0]
    j_stop = num_unit_cells[1] * shape[1]
    i, j = tuple(
        jnp.meshgrid(
            jnp.arange(0.5, i_stop) / shape[0],
            jnp.arange(0.5, j_stop) / shape[1],
            indexing="ij",
        )
    )
    x = (
        i * primitive_lattice_vectors.u[..., jnp.newaxis, jnp.newaxis, 0]
        + j * primitive_lattice_vectors.v[..., jnp.newaxis, jnp.newaxis, 0]
    )
    y = (
        i * primitive_lattice_vectors.u[..., jnp.newaxis, jnp.newaxis, 1]
        + j * primitive_lattice_vectors.v[..., jnp.newaxis, jnp.newaxis, 1]
    )
    return x, y


# -----------------------------------------------------------------------------
# Functions related to transverse wavevectors.
# -----------------------------------------------------------------------------


def plane_wave_in_plane_wavevector(
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    permittivity: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the in-plane wavevector for a plane-wave excitation.

    Args:
        wavelength: The free-space wavelength of the plane-wave excitation.
        polar_angle: Polar angle of the plane-wave excitation.
        azimuthal_angle: Azimuthal angle of plane-wave the excitation.
        permittivity: Scalar permittivity of the medium in which the polar angle
            and azimuthal angle are specified.

    Returns:
        The fundamental transverse wavevector, i.e. ``(kx0, ky0)``.
    """
    angular_frequency = utils.angular_frequency_for_wavelength(wavelength)
    kx0 = (
        angular_frequency
        * jnp.sin(polar_angle)
        * jnp.cos(azimuthal_angle)
        * jnp.sqrt(permittivity.real)
    )
    ky0 = (
        angular_frequency
        * jnp.sin(polar_angle)
        * jnp.sin(azimuthal_angle)
        * jnp.sqrt(permittivity.real)
    )
    return jnp.stack([kx0, ky0], axis=-1)


def transverse_wavevectors(
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
) -> jnp.ndarray:
    """Obtains all the relevant transverse wavevectors given the expansion.

    Args:
        in_plane_wavevector: The zeroth-order transverse wavevector.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        expansion: The expansion for which the set of transverse wavevectors are sought.

    Returns:
        The transverse wavevectors.
    """
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    kx = in_plane_wavevector[..., 0, jnp.newaxis] + 2 * jnp.pi * (
        expansion.basis_coefficients[:, 0] * reciprocal_vectors.u[..., 0, jnp.newaxis]
        + expansion.basis_coefficients[:, 1] * reciprocal_vectors.v[..., 0, jnp.newaxis]
    )
    ky = in_plane_wavevector[..., 1, jnp.newaxis] + 2 * jnp.pi * (
        expansion.basis_coefficients[:, 0] * reciprocal_vectors.u[..., 1, jnp.newaxis]
        + expansion.basis_coefficients[:, 1] * reciprocal_vectors.v[..., 1, jnp.newaxis]
    )
    batch_shape = jnp.broadcast_shapes(
        in_plane_wavevector.shape[:-1],
        primitive_lattice_vectors.u.shape[:-1],
    )
    assert kx.shape == batch_shape + (expansion.num_terms,)
    return jnp.stack([kx, ky], axis=-1)


# -----------------------------------------------------------------------------
# Functions to generate basis coefficients with various truncations.
# -----------------------------------------------------------------------------


def _basis_coefficients_circular(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
) -> onp.ndarray:
    """Computes the basis coefficients of lattice vectors lying within a circle.

    The coefficients generate a set of lattice vectors from a basis using a circular
    truncation, so that all lattice vectors lying within a circular region with area
    given by ``num * u ⨉ v`` are included, where ``⨉`` is the cross product. The size of
    the set will generally be close to `approximate_num_terms`.

    Args:
        primitive_lattice_vectors: Primitive vectors for the reciprocal-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a symmetric expansion, the total number of terms may differ from
            this value.

    Returns:
        The coefficients, with shape ``(num_vectors, 2)``. The final axis gives the
        coefficient for the first and second vector in the basis.
    """
    # Generate candidate coefficients. These will include more coefficients than needed;
    # subsequently we will filter based on magnitude.
    g = onp.arange(-approximate_num_terms // 2, approximate_num_terms // 2 + 1)
    G1, G2 = onp.meshgrid(g, g, indexing="ij")
    G1 = G1.flatten()
    G2 = G2.flatten()

    # Generate the actual vectors and compute their magnitude.
    vectors = (
        G1[..., onp.newaxis] * primitive_lattice_vectors.u[..., onp.newaxis, :]
        + G2[..., onp.newaxis] * primitive_lattice_vectors.v[..., onp.newaxis, :]
    )
    magnitude = onp.linalg.norm(vectors, axis=-1)

    # Include all vectors lying within the circle with area equal to cross product
    # of `u` and `v` scaled by `num_terms`.
    max_magnitude = onp.sqrt(
        approximate_num_terms
        * onp.abs(
            _cross_product(primitive_lattice_vectors.u, primitive_lattice_vectors.v)
        )
        / onp.pi
    )
    mask = magnitude < max_magnitude

    G1 = G1[mask]
    G2 = G2[mask]
    magnitude = magnitude[mask]
    G = onp.stack([G1, G2], axis=-1)

    order = onp.argsort(magnitude, kind="stable")
    return G[..., order, :]


def _basis_coefficients_parallelogramic(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
) -> onp.ndarray:
    """Computes the basis coefficients of lattice vectors lying within a parallelogram.

    The coefficients generate a set of lattice vectors from a basis using a
    parallelogramic truncation. The size of the set will generally be close to
    ``approximate_num_terms``.

    Args:
        primitive_lattice_vectors: Primitive vectors for the reciprocal-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a full parallelogram expansion, the total number of terms may
            differ from this value.

    Returns:
        The coefficients, with shape ``(num_vectors, 2)``. The final axis gives the
        coefficient for the first and second vector in the basis.
    """

    ku_spacing = onp.sqrt(onp.sum(onp.abs(primitive_lattice_vectors.u) ** 2))
    kv_spacing = onp.sqrt(onp.sum(onp.abs(primitive_lattice_vectors.v) ** 2))

    # Solve for `(nu, nv)` such that we approximately satisfy
    #     (2 * nu + 1) * (2 * nv + 1) = approximate_num_terms
    # and
    #     nu / nv = kv_spacing / ku_spacing
    # Since `ku_spacing` and `kv_spacing` give the spacing of points in reciprocal
    # space along the ku and kv directions, respectively, this second equation ensures
    # that we are chosing points in a parallelogram with equal-length sides in k-space.
    # (Note that while the sides of the parallelogram have equal length, the number of
    # points along each direction will differ, as these are spaced by `ku_spacing` and
    # `kv_spacing`.)

    def _solve_quadratic(ratio):
        a = 4 * ratio
        b = 2 * (ratio + 1)
        c = 1 - approximate_num_terms
        nu = (-b + onp.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return int(onp.around(nu))

    nu = _solve_quadratic(ku_spacing / kv_spacing)
    nv = _solve_quadratic(kv_spacing / ku_spacing)

    G1, G2 = onp.meshgrid(
        onp.arange(-nu, nu + 1),
        onp.arange(-nv, nv + 1),
        indexing="ij",
    )
    G1 = G1.flatten()
    G2 = G2.flatten()
    G = onp.stack([G1, G2], axis=-1)
    # Generate the actual vectors and compute their magnitude.
    vectors = (
        G1[..., onp.newaxis] * primitive_lattice_vectors.u[..., onp.newaxis, :]
        + G2[..., onp.newaxis] * primitive_lattice_vectors.v[..., onp.newaxis, :]
    )
    magnitude = onp.linalg.norm(vectors, axis=-1)

    order = onp.argsort(magnitude, kind="stable")
    return G[..., order, :]


def _cross_product(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the cross product of 2D vectors ``x`` and ``y``."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


tree_util.register_pytree_node(
    LatticeVectors,
    lambda lv: ((lv.u, lv.v), None),
    lambda _, uv: LatticeVectors(*uv),
)


tree_util.register_pytree_node(
    Truncation,
    lambda x: ((), x.value),
    lambda value, _: Truncation(value),
)


def _unflatten_expansion(
    aux_tuple: Tuple["_HashableArray"],
    leaves: Tuple,
) -> Expansion:
    """Unflattens the `Expansion`."""
    del leaves
    wrapped: "_HashableArray" = aux_tuple[0]
    coeffs: onp.ndarray = wrapped.value
    return Expansion(basis_coefficients=coeffs)


tree_util.register_pytree_node(
    Expansion,
    lambda e: ((), (_HashableArray(e.basis_coefficients),)),
    unflatten_func=_unflatten_expansion,
)


class _HashableArray:
    """Hashable wrapper for numpy arrays."""

    def __init__(self, value: onp.ndarray):
        self.value: onp.ndarray = value

    def __hash__(self):
        return hash((self.value.shape, self.value.dtype, self.value.tobytes()))

    def __eq__(self, other):
        if not isinstance(other, _HashableArray):
            return False
        return (
            self.value.shape == other.value.shape
            and self.value.dtype == other.value.dtype
            and onp.all(self.value == other.value)
        )
