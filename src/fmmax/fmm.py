"""Functions related to layer eigenmode calculation for the FMM algorithm.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import enum
import functools
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from fmmax import basis, eig, fft, fmm_matrices, misc, utils, vector

# xx, xy, yx, yy, and zz components of permittivity or permeability.
TensorComponents = Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]
VectorFn = Callable[
    [jnp.ndarray, basis.Expansion, basis.LatticeVectors],
    Tuple[jnp.ndarray, jnp.ndarray],
]


@enum.unique
class Formulation(enum.Enum):
    """Enumerates supported Fourier modal method formulations.

    Each formulation specifies an algorithm to compute the transverse permittivity
    matrix used in the Fourier modal method. The simplest formulation is ``FFT``, in
    which the blocks of the transverse permittivity matrix are simply the Fourier
    convolution matrices for their respective permittivity tensor components.

    The remaining formulations are so-called vector formulations, which make use of a
    vector field generated in the unit cell of the FMM calculation. The vector field
    defines a local coordinate system that tangent and normal to the interfaces of
    features in the unit cell, allowing improved convergence through independent
    treatment of field components that are tangent and normal to the interfaces.
    """

    #: The simplest formulation, which does not consider the orientation of
    #: interfaces of features in a permittivity array.
    FFT = "fft"

    #: Generates a complex linear vector field which has maximum magnitude ``1`` and
    #: a null in the interior of features. In the objective, the gradient of the
    #: vector field on the real-space grid is computed; a penalty term discourages
    #: non-smooth fields (i.e. fields whose gradient is large).
    POL = vector.POL

    #: Takes the field computed by ``POL`` and normalizes so the magnitude is ``1``
    #: evereywhere in the unit cell. Where ``POL`` has zeros, ``NORMAL`` has
    #: discontinuities.
    NORMAL = vector.NORMAL

    #: Takes the field computed by ``POL``, and converts it to a complex
    #: elliptical field which has magnitude ``1`` everywhere and lacks discontinuities.
    JONES = vector.JONES

    #: Directly computes a complex elliptical vector field without first finding
    #: a linear vector field. Smoothness is evaluated on the real-space grid.
    JONES_DIRECT = vector.JONES_DIRECT

    #: Generates a complex linear vector field, but with an alternate method of
    #: penalizing non-smoothness. Specifically, the Fourier components corresponding to
    #: high spatial frequencies are penalized. Compared to ``POL``, ``POL_FOURIER``
    #: can be computed more efficiently.
    POL_FOURIER = vector.POL_FOURIER

    #: Takes the field computed by ``POL_FOURIER``and normalizes so the magnitude is
    #: ``1`` evereywhere in the unit cell.
    NORMAL_FOURIER = vector.NORMAL_FOURIER

    #: Takes the field computed by ``POL_FOURIER`` and converts it to a complex
    #: elliptical field.
    JONES_FOURIER = vector.JONES_FOURIER

    #: Directly computes a complex elliptical vector field, using Fourier coefficients
    #: to penalize non-smoothness.
    JONES_DIRECT_FOURIER = vector.JONES_DIRECT_FOURIER


_DEFAULT_FORMULATION = Formulation.JONES_DIRECT_FOURIER


def eigensolve_isotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn = _DEFAULT_FORMULATION,
) -> "LayerSolveResult":
    """Performs the eigensolve for a layer with isotropic permittivity.

    This function performs either a uniform-layer or patterned-layer eigensolve,
    depending on the shape of the trailing dimensions of a given layer permittivity.
    When the final two dimensions have shape ``(1, 1)``, the layer is treated as
    uniform. Otherwise, it is patterned.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape ``(..., nx, ny)``.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, or a callable which computes
            the tangent vector field for a custom vector FMM formulation. The default
            is ``Formulation.JONES_DIRECT_FOURIER``.

    Returns:
        The ``LayerSolveResult``.
    """
    if permittivity.ndim < 2:
        raise ValueError(
            f"`permittivity` must have at least two dimensions, with the final two"
            f"corresponding to the spatial dimensions of the unit cell, but got shape "
            f"{permittivity.shape}."
        )
    if permittivity.shape[-2:] == (1, 1):
        _eigensolve_fn = _eigensolve_uniform_isotropic_media
    else:
        _eigensolve_fn = functools.partial(
            _eigensolve_patterned_isotropic_media, formulation=formulation
        )

    return _eigensolve_fn(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=permittivity,
        expansion=expansion,
    )


def eigensolve_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    permittivity_zz: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn = _DEFAULT_FORMULATION,
) -> "LayerSolveResult":
    """Performs the eigensolve for a layer with anisotropic permittivity.

    This function performs either a uniform-layer or patterned-layer eigensolve,
    depending on the shape of the trailing dimensions of a given layer permittivity.
    When the final two dimensions have shape ``(1, 1)``, the layer is treated as
    uniform. Otherwise, it is patterned.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape ``(..., nx, ny)``.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        permittivity_zz: The zz-component of the permittivity tensor.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, or a callable which computes
            the tangent vector field for a custom vector FMM formulation.

    Returns:
        The ``LayerSolveResult``.
    """
    shapes = tuple(
        p.shape
        for p in [
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
        ]
    )
    if not all(shape == shapes[0] for shape in shapes) and len(shapes[0]) > 2:
        raise ValueError(
            f"Permittivities must have matching shapes and at least two dimensions, "
            f"with the final two corresponding to the spatial dimensions of the unit "
            f"cell, but got shapes {shapes}."
        )
    return eigensolve_general_anisotropic_media(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity_xx=permittivity_xx,
        permittivity_xy=permittivity_xy,
        permittivity_yx=permittivity_yx,
        permittivity_yy=permittivity_yy,
        permittivity_zz=permittivity_zz,
        permeability_xx=jnp.ones_like(permittivity_xx),
        permeability_xy=jnp.zeros_like(permittivity_xx),
        permeability_yx=jnp.zeros_like(permittivity_xx),
        permeability_yy=jnp.ones_like(permittivity_xx),
        permeability_zz=jnp.ones_like(permittivity_xx),
        expansion=expansion,
        formulation=formulation,
        vector_field_source=None,
    )


def eigensolve_general_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    permittivity_zz: jnp.ndarray,
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    permeability_zz: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn = _DEFAULT_FORMULATION,
    vector_field_source: Optional[jnp.ndarray] = None,
) -> "LayerSolveResult":
    """Performs the eigensolve for a general anistropic layer.

    Here, "general" refers to the fact that the layer material can be magnetic, i.e.
    the permeability and permittivity can be specified.

    This function performs either a uniform-layer or patterned-layer eigensolve,
    depending on the shape of the trailing dimensions of a given layer permittivity.
    When the final two dimensions have shape ``(1, 1)``, the layer is treated as
    uniform. Otherwise, it is patterned.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity_xx: The xx-component of the permittivity tensor, with
            shape ``(..., nx, ny)``.
        permittivity_xy: The xy-component of the permittivity tensor.
        permittivity_yx: The yx-component of the permittivity tensor.
        permittivity_yy: The yy-component of the permittivity tensor.
        permittivity_zz: The zz-component of the permittivity tensor.
        permeability_xx: The xx-component of the permeability tensor.
        permeability_xy: The xy-component of the permeability tensor.
        permeability_yx: The yx-component of the permeability tensor.
        permeability_yy: The yy-component of the permeability tensor.
        permeability_zz: The zz-component of the permeability tensor.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, or a callable which computes
            the tangent vector field for a custom vector FMM formulation.
        vector_field_source: Optional array used to calculate the vector field for
            vector formulations of the FMM. If not specified, ``(permittivity_xx +
            permittivity_yy) / 2`` is used. Ignored for the ``FFT`` formulation. Should
            have shape matching the permittivities and permeabilities.

    Returns:
        The ``LayerSolveResult``.
    """
    shapes = tuple(
        p.shape
        for p in [
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
        ]
    )
    if not all(shape == shapes[0] for shape in shapes) and len(shapes[0]) > 2:
        raise ValueError(
            f"Permittivities must have matching shapes and at least two dimensions, "
            f"with the final two corresponding to the spatial dimensions of the unit "
            f"cell, but got shapes {shapes}."
        )
    if permittivity_xx.shape[-2:] == (1, 1):
        _eigensolve_fn = _eigensolve_uniform_general_anisotropic_media
    else:
        if vector_field_source is None:
            vector_field_source = (permittivity_xx + permittivity_yy) / 2
        _eigensolve_fn = functools.partial(
            _eigensolve_patterned_general_anisotropic_media,
            formulation=formulation,
            vector_field_source=vector_field_source,
        )

    return _eigensolve_fn(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivities=(
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
        ),
        permeabilities=(
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
        ),
        expansion=expansion,
    )


@dataclasses.dataclass
class LayerSolveResult:
    """Stores the result of a layer eigensolve.

    This eigenvalue problem is specified in equation 28 of [2012 Liu].

    Attributes:
        wavelength: The wavelength for the solve.
        in_plane_wavevector: The in-plane wavevector for the solve.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        expansion: The expansion used for the eigensolve.
        eigenvalues: The layer eigenvalues.
        eigenvectors: The layer eigenvectors.
        z_permittivity_matrix: The fourier-transformed zz-component of permittivity.
        inverse_z_permittivity_matrix: The fourier-transformed inverse of zz-component
            of permittivity.
        z_permeability_matrix: The fourier-transformed zz-component of permeability.
        inverse_z_permeability_matrix: The fourier-transformed inverse of zz-component
            of permeability.
        transverse_permeability_matrix: The transverse permeability matrix, needed to
            calculate the omega-script-k matrix from equation 26 of [2012 Liu]. This
            is needed to generate the layer scattering matrix.
        tangent_vector_field: The tangent vector field ``(tx, ty)`` used to compute the
            transverse permittivity matrix, if a vector FMM formulation is used. If
            the ``FFT`` formulation is used, the vector field is ``None``.
    """

    wavelength: jnp.ndarray
    in_plane_wavevector: jnp.ndarray
    primitive_lattice_vectors: basis.LatticeVectors
    expansion: basis.Expansion
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    z_permittivity_matrix: jnp.ndarray
    inverse_z_permittivity_matrix: jnp.ndarray
    z_permeability_matrix: jnp.ndarray
    inverse_z_permeability_matrix: jnp.ndarray
    transverse_permeability_matrix: jnp.ndarray
    tangent_vector_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]]

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self.eigenvectors.shape[:-2]

    @property
    def omega_script_k_matrix(self):
        """Compute omega-script-k matrix from equation 26 of [2012 Liu]."""
        return fmm_matrices.omega_script_k_matrix_patterned(
            wavelength=self.wavelength,
            z_permittivity_matrix=self.z_permittivity_matrix,
            transverse_permeability_matrix=self.transverse_permeability_matrix,
            transverse_wavevectors=basis.transverse_wavevectors(
                in_plane_wavevector=self.in_plane_wavevector,
                primitive_lattice_vectors=self.primitive_lattice_vectors,
                expansion=self.expansion,
            ),
        )

    def __post_init__(self) -> None:
        """Validates shapes of the ``LayerSolveResult`` attributes."""
        # Avoid validation when attributes are e.g. tracers.
        if not isinstance(self.eigenvalues, (jnp.ndarray, onp.ndarray)):
            return

        if self.wavelength.shape != self.batch_shape:
            raise ValueError(
                f"`wavelength` must have shape equal to the batch shape, but got "
                f"{self.wavelength.shape} when the batch shape is "
                f"{self.batch_shape}."
            )
        if self.in_plane_wavevector.shape != self.batch_shape + (2,):
            raise ValueError(
                f"`in_plane_wavevector` must have leading dimensions with shape "
                f"equal to the batch shape and a trailing dimension of size 2,"
                f"but got {self.in_plane_wavevector.shape} when the batch shape is "
                f"{self.batch_shape}."
            )
        if self.primitive_lattice_vectors.u.shape != self.batch_shape + (
            2,
        ) or self.primitive_lattice_vectors.u.shape != self.batch_shape + (2,):
            raise ValueError(
                f"`primitive_lattice_vectors` must each have leading dimensions with "
                f"shape equal to the batch shape and a trailing dimension of size 2,"
                f"but got {self.primitive_lattice_vectors.u.shape} and "
                f"{self.primitive_lattice_vectors.v.shape} when the batch shape is "
                f"{self.batch_shape}."
            )

        if self.expansion.num_terms * 2 != self.eigenvectors.shape[-1]:
            raise ValueError(
                f"`eigenvectors` must have shape compatible with `expansion.num_terms`,"
                f" but got shape {self.eigenvectors.shape} when `num_terms` shape is "
                f"{self.expansion.num_terms}."
            )
        if self.eigenvalues.shape != self.eigenvectors.shape[:-1]:
            raise ValueError(
                f"`eigenvalues` must have compatible shape, but got shape "
                f"{self.eigenvalues.shape} when `eigenvectors` shape is "
                f"{self.eigenvectors.shape}."
            )

        expected_matrix_shape = self.batch_shape + (self.expansion.num_terms,) * 2

        if self.inverse_z_permittivity_matrix.shape != expected_matrix_shape:
            raise ValueError(
                f"`inverse_z_permittivity_matrix` must have shape "
                f"{expected_matrix_shape} when `eigenvectors` has shape "
                f"{self.eigenvectors.shape}, but got "
                f"{self.inverse_z_permittivity_matrix.shape}."
            )
        if self.z_permittivity_matrix.shape != expected_matrix_shape:
            raise ValueError(
                f"`z_permittivity_matrix` must have shape "
                f"{expected_matrix_shape} when `eigenvectors` has shape "
                f"{self.eigenvectors.shape}, but got "
                f"{self.z_permittivity_matrix.shape}."
            )
        if self.inverse_z_permeability_matrix.shape != expected_matrix_shape:
            raise ValueError(
                f"`inverse_z_permeability_matrix` must have shape "
                f"{expected_matrix_shape} when `eigenvectors` has shape "
                f"{self.eigenvectors.shape}, but got "
                f"{self.inverse_z_permeability_matrix.shape}."
            )
        if self.z_permeability_matrix.shape != expected_matrix_shape:
            raise ValueError(
                f"`z_permeability_matrix` must have shape "
                f"{expected_matrix_shape} when `eigenvectors` has shape "
                f"{self.eigenvectors.shape}, but got "
                f"{self.z_permeability_matrix.shape}."
            )
        if self.transverse_permeability_matrix.shape != self.eigenvectors.shape:
            raise ValueError(
                f"`transverse_permeability_matrix` must have shape matching that of "
                f"`eigenvectors`, but got shapes "
                f"{self.transverse_permeability_matrix.shape} and "
                f"{self.eigenvectors.shape}."
            )

        if self.tangent_vector_field is not None and (
            self.tangent_vector_field[0].ndim != self.eigenvectors.ndim
        ):
            raise ValueError(
                f"`tangent_vector_field` must have ndim compatible with "
                f"`eigenvectors`, but got shapes {self.tangent_vector_field[0]} and "
                f"{self.eigenvectors}."
            )


# -----------------------------------------------------------------------------
# Eigensolves for specific cases, e.g. uniform isotropic, anisotropic, etc.
# -----------------------------------------------------------------------------


def _eigensolve_uniform_isotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
) -> LayerSolveResult:
    r"""Returns the the results of a uniform isotropic layer eigensolve.

    The layer is uniform and isotropic, in the sense that the permittivity does not
    vary spatially and has no orientation dependence. In this case, the eigenvalues
    and eigenvectors can be calculated analytically.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The scalar permittivity for the layer, with shape ``(..., 1, 1)``.
        expansion: The field expansion to be used.

    Returns:
        The ``LayerSolveResult``.
    """
    (
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        (permittivity,),
    ) = _validate_and_broadcast(
        wavelength, in_plane_wavevector, primitive_lattice_vectors, permittivity
    )
    if permittivity.shape[-2:] != (1, 1):
        raise ValueError(
            f"Trailing axes of `permittivity` must have shape (1, 1) but got a shape "
            f"of {permittivity.shape}."
        )

    batch_shape = jnp.broadcast_shapes(
        wavelength.shape, in_plane_wavevector.shape[:-1], permittivity.shape[:-2]
    )

    num_eigenvalues = 2 * expansion.num_terms
    permittivity = jnp.squeeze(permittivity, axis=(-2, -1))

    # Transverse wavevectors are the `k + G` from equation 5 of [2012 Liu].
    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )

    # In uniform media, the eigenvectors are just the plane waves.
    dtype = jnp.promote_types(jnp.complex64, permittivity.dtype)
    eigenvectors = jnp.broadcast_to(
        jnp.eye(num_eigenvalues, dtype=dtype),
        shape=batch_shape + (num_eigenvalues, num_eigenvalues),
    )

    angular_frequency = utils.angular_frequency_for_wavelength(wavelength)
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    eigenvalues = jnp.sqrt(
        (
            permittivity[..., jnp.newaxis] * angular_frequency[..., jnp.newaxis] ** 2
            - kx**2
            - ky**2
        ).astype(dtype)
    )
    eigenvalues = _select_eigenvalues_sign(eigenvalues)
    eigenvalues = jnp.tile(eigenvalues, 2)

    diag_shape = permittivity.shape + (expansion.num_terms,)
    inverse_z_permittivity_diag = jnp.broadcast_to(
        1 / permittivity[..., jnp.newaxis], diag_shape
    )
    inverse_z_permittivity_matrix = misc.diag(inverse_z_permittivity_diag)

    z_permittivity_diag = jnp.broadcast_to(permittivity[..., jnp.newaxis], diag_shape)
    z_permittivity_matrix = misc.diag(z_permittivity_diag)
    z_permeability_matrix = misc.diag(jnp.ones(diag_shape, dtype=dtype))

    transverse_diag_shape = permittivity.shape + (2 * expansion.num_terms,)
    transverse_permeability_matrix = misc.diag(
        jnp.ones(transverse_diag_shape, dtype=dtype)
    )

    return LayerSolveResult(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        tangent_vector_field=None,
    )


def _eigensolve_patterned_isotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn,
) -> LayerSolveResult:
    r"""Returns the results of a patterned isotropic layer eigensolve.

    The layer is patterned and isotropic, in the sense that the permittivity varies
    spatially and has no orientation dependence.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, or a callable which computes
            the tangent vector field for a custom vector FMM formulation.

    Returns:
        The ``LayerSolveResult``.
    """
    (
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        (permittivity,),
    ) = _validate_and_broadcast(
        wavelength, in_plane_wavevector, primitive_lattice_vectors, permittivity
    )
    (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        tangent_vector_field,
    ) = _fourier_matrices_patterned_isotropic_media(
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivity=permittivity,
        expansion=expansion,
        formulation=formulation,
    )

    # Create permeability matrices for nonmagnetic materials.
    ones = jnp.ones(
        z_permittivity_matrix.shape[:-1],
        dtype=z_permittivity_matrix.dtype,
    )
    zeros = jnp.zeros_like(ones)
    z_permeability_matrix = misc.diag(ones)
    inverse_z_permeability_matrix = misc.diag(ones)
    transverse_permeability_matrix = jnp.block(
        [
            [misc.diag(ones), misc.diag(zeros)],
            [misc.diag(zeros), misc.diag(ones)],
        ]
    )

    return _numerical_eigensolve(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        transverse_permittivity_matrix=transverse_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        expansion=expansion,
        tangent_vector_field=tangent_vector_field,
    )


def _eigensolve_uniform_general_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: TensorComponents,
    permeabilities: TensorComponents,
    expansion: basis.Expansion,
) -> LayerSolveResult:
    """Returns the results of a uniform anisotropic layer eigensolve.

    The layer is uniform and anisotropic, in the sense that the permittivity does not
    vary spatially and has orientation dependence.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: ``(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)``, each having shape ``(..., nx, ny)``.
        permeabilities: The elements of the permeability tensor: ``(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)``, each having shape ``(..., nx, ny)``.
        expansion: The field expansion to be used.

    Returns:
        The ``LayerSolveResult``.
    """
    if not all([p.shape[-2:] == (1, 1) for p in permittivities + permeabilities]):
        raise ValueError(
            f"Trailing axes of arrays in `permittivities` and `permeabilities` must "
            f"have shape (1, 1) but got a shapes {[p.shape for p in permittivities]} "
            f"and {[p.shape for p in permeabilities]}."
        )
    (
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        (
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
        ),
    ) = _validate_and_broadcast(
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        *permittivities,
        *permeabilities,
    )

    _ = jnp.broadcast_shapes(
        wavelength.shape, in_plane_wavevector.shape[:-1], permittivity_xx.shape[:-2]
    )

    shape = permittivity_xx.shape[:-2] + (expansion.num_terms,)
    permittivity_xx = jnp.broadcast_to(jnp.squeeze(permittivity_xx, axis=-1), shape)
    permittivity_xy = jnp.broadcast_to(jnp.squeeze(permittivity_xy, axis=-1), shape)
    permittivity_yx = jnp.broadcast_to(jnp.squeeze(permittivity_yx, axis=-1), shape)
    permittivity_yy = jnp.broadcast_to(jnp.squeeze(permittivity_yy, axis=-1), shape)
    permittivity_zz = jnp.broadcast_to(jnp.squeeze(permittivity_zz, axis=-1), shape)
    z_permittivity_matrix = misc.diag(permittivity_zz)
    inverse_z_permittivity_matrix = misc.diag(1 / permittivity_zz)
    # Note that the matrix element ordering and signs differ from [2012 Liu]
    # equation 37, but are consistent with the definition in equation 15. Equation 37
    # is likely in error.
    transverse_permittivity_matrix = jnp.block(
        [
            [misc.diag(permittivity_yy), misc.diag(-permittivity_yx)],
            [misc.diag(-permittivity_xy), misc.diag(permittivity_xx)],
        ]
    )

    permeability_xx = jnp.broadcast_to(jnp.squeeze(permeability_xx, axis=-1), shape)
    permeability_xy = jnp.broadcast_to(jnp.squeeze(permeability_xy, axis=-1), shape)
    permeability_yx = jnp.broadcast_to(jnp.squeeze(permeability_yx, axis=-1), shape)
    permeability_yy = jnp.broadcast_to(jnp.squeeze(permeability_yy, axis=-1), shape)
    permeability_zz = jnp.broadcast_to(jnp.squeeze(permeability_zz, axis=-1), shape)
    z_permeability_matrix = misc.diag(permeability_zz)
    inverse_z_permeability_matrix = misc.diag(1 / permeability_zz)
    # Note that the matrix element ordering for the transverse permittivity and
    # permeability matrices differs.
    transverse_permeability_matrix = jnp.block(
        [
            [misc.diag(permeability_xx), misc.diag(permeability_xy)],
            [misc.diag(permeability_yx), misc.diag(permeability_yy)],
        ]
    )

    return _numerical_eigensolve(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        transverse_permittivity_matrix=transverse_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        expansion=expansion,
        tangent_vector_field=None,
    )


def _eigensolve_patterned_general_anisotropic_media(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: TensorComponents,
    permeabilities: TensorComponents,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn,
    vector_field_source: jnp.ndarray,
) -> LayerSolveResult:
    """Returns the results of a patterned anisotropic layer eigensolve.

    The layer is patterned and anisotropic, in the sense that the permittivity varies
    spatially and has orientation dependence.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: ``(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)``, each having shape ``(..., nx, ny)``.
        permeabilities: The elements of the permeability tensor: ``(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)``, each having shape ``(..., nx, ny)``.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, or a callable which computes
            the tangent vector field for a custom vector FMM formulation.
        vector_field_source: Array used to calculate the vector field, with shape
            matching the permittivities and permeabilities.

    Returns:
        The ``LayerSolveResult``.
    """
    (
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        (
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
            vector_field_source,
        ),
    ) = _validate_and_broadcast(
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        *permittivities,
        *permeabilities,
        vector_field_source,
    )
    (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        inverse_z_permeability_matrix,
        z_permeability_matrix,
        transverse_permeability_matrix,
        tangent_vector_field,
    ) = _fourier_matrices_patterned_anisotropic_media(
        primitive_lattice_vectors=primitive_lattice_vectors,
        permittivities=(
            permittivity_xx,
            permittivity_xy,
            permittivity_yx,
            permittivity_yy,
            permittivity_zz,
        ),
        permeabilities=(
            permeability_xx,
            permeability_xy,
            permeability_yx,
            permeability_yy,
            permeability_zz,
        ),
        expansion=expansion,
        formulation=formulation,
        vector_field_source=vector_field_source,
    )
    return _numerical_eigensolve(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        transverse_permittivity_matrix=transverse_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        expansion=expansion,
        tangent_vector_field=tangent_vector_field,
    )


# -----------------------------------------------------------------------------
# Helper function used by all eigensolves done numerically.
# -----------------------------------------------------------------------------


def _numerical_eigensolve(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    z_permittivity_matrix: jnp.ndarray,
    inverse_z_permittivity_matrix: jnp.ndarray,
    transverse_permittivity_matrix: jnp.ndarray,
    z_permeability_matrix: jnp.ndarray,
    inverse_z_permeability_matrix: jnp.ndarray,
    transverse_permeability_matrix: jnp.ndarray,
    expansion: basis.Expansion,
    tangent_vector_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
) -> LayerSolveResult:
    r"""Returns the results of a patterned layer eigensolve.

    The layer may be anisotropic and magnetic, as determined by the provided transverse
    permittivity and permeability matrices.

    Args:
        wavelength: The free space wavelength of the excitation.
        in_plane_wavevector: ``(kx0, ky0)``.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        inverse_z_permittivity_matrix: The fourier-transformed inverse of zz-component
            of permittivity.
        z_permittivity_matrix: The fourier-transformed zz-component of permittivity.
        transverse_permittivity_matrix: The fourier-transformed transverse permittivity
            matrix from equation 15 of [2012 Liu].
        inverse_z_permeability_matrix: The fourier-transformed inverse of zz-component
            of permeability.
        z_permeability_matrix: The fourier-transformed zz-component of permeability.
        transverse_permeability_matrix: The fourier-transformed transverse permeability
            matrix.
        expansion: The field expansion to be used.
        tangent_vector_field: The tangent vector field ``(tx, ty)`` used to compute the
            transverse permittivity matrix, if a vector FMM formulation is used. If
            the ``FFT`` formulation is used, the vector field is ``None``.

    Returns:
        The ``LayerSolveResult``.
    """
    # Transverse wavevectors are the `k + G` from equation 5 of [2012 Liu].
    transverse_wavevectors = basis.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )

    # The k matrix from equation 23 of [2012 Liu], modified for magnetic materials.
    k_matrix = fmm_matrices.k_matrix_patterned(
        z_permeability_matrix, transverse_wavevectors
    )

    omega_script_k_matrix = fmm_matrices.omega_script_k_matrix_patterned(
        wavelength=wavelength,
        z_permittivity_matrix=z_permittivity_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        transverse_wavevectors=transverse_wavevectors,
    )

    # The matrix from equation 28 of [2012 Liu], modified for magnetic materials.
    matrix = (
        transverse_permittivity_matrix @ omega_script_k_matrix
        - k_matrix @ transverse_permeability_matrix
    )
    eigenvalues_squared, eigenvectors = eig.eig(matrix)
    eigenvalues = jnp.sqrt(eigenvalues_squared)
    eigenvalues = _select_eigenvalues_sign(eigenvalues)
    return LayerSolveResult(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        z_permittivity_matrix=z_permittivity_matrix,
        inverse_z_permittivity_matrix=inverse_z_permittivity_matrix,
        z_permeability_matrix=z_permeability_matrix,
        inverse_z_permeability_matrix=inverse_z_permeability_matrix,
        transverse_permeability_matrix=transverse_permeability_matrix,
        tangent_vector_field=tangent_vector_field,
    )


# -----------------------------------------------------------------------------
# Functions for computing Fourier convolution matrices .
# -----------------------------------------------------------------------------


def _fourier_matrices_patterned_isotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivity: jnp.ndarray,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]
]:
    """Return Fourier convolution matrices for patterned nonmagnetic isotropic media.

    All matrices are forms of the Fourier convolution matrices defined in equation
    8 of [2012 Liu]. For vector formulations, the transverse permittivity matrix is
    of the form E2 given in equation 51 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivity: The permittivity array, with shape ``(..., nx, ny)``.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used, or a callable which computes
            the tangent vector field for a custom vector FMM formulation.

    Returns:
        inverse_z_permittivity_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix.
        tangent_vector_field: The tangent vector field ``(tx, ty)`` used to compute the
            transverse permittivity matrix, if a vector FMM formulation is used. If
            the ``FFT`` formulation is used, the vector field is ``None``.
    """
    if formulation is Formulation.FFT:
        _transverse_permittivity_fn = functools.partial(
            fmm_matrices.transverse_permittivity_fft,
            expansion=expansion,
        )
        tangent_vector_field = None
    else:
        if isinstance(formulation, Formulation):
            vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
        else:
            vector_fn = formulation
        tx, ty = vector_fn(permittivity, expansion, primitive_lattice_vectors)
        _transverse_permittivity_fn = functools.partial(
            fmm_matrices.transverse_permittivity_vector,
            tx=tx,
            ty=ty,
            expansion=expansion,
        )
        tangent_vector_field = (tx, ty)

    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    inverse_z_permittivity_matrix = _transform(1 / permittivity)
    z_permittivity_matrix = _transform(permittivity)
    transverse_permittivity_matrix = _transverse_permittivity_fn(permittivity)

    return (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        tangent_vector_field,
    )


def _fourier_matrices_patterned_anisotropic_media(
    primitive_lattice_vectors: basis.LatticeVectors,
    permittivities: TensorComponents,
    permeabilities: TensorComponents,
    expansion: basis.Expansion,
    formulation: Formulation | VectorFn,
    vector_field_source: jnp.ndarray,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Optional[Tuple[jnp.ndarray, jnp.ndarray]],
]:
    """Return Fourier convolution matrices for patterned anisotropic media.

    The transverse permittivity matrix E is defined as,

        [-Dy, Dx]^T = E [-Ey, Ex]^T

    while the transverse permeability matrix M is defined as,

        [Bx, By]^T = M [Hx, Hy]^T

    The Fourier factorization is done as for E1 given in equation 47 of [2012 Liu].

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        permittivities: The elements of the permittivity tensor: ``(eps_xx, eps_xy,
            eps_yx, eps_yy, eps_zz)``, each having shape ``(..., nx, ny)``.
        permeabilities: The elements of the permeability tensor: ``(mu_xx, mu_xy,
            mu_yx, mu_yy, mu_zz)``, each having shape ``(..., nx, ny)``.
        expansion: The field expansion to be used.
        formulation: Specifies the formulation to be used.
        vector_field_source: Array used to calculate the vector field, with shape
            matching the permittivities and permeabilities.

    Returns:
        inverse_z_permittivity_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permittivity.
        z_permittivity_matrix: The Fourier convolution matrix for the z-component
            of the permittivity.
        transverse_permittivity_matrix: The transverse permittivity matrix from
            equation 15 of [2012 Liu], computed in the manner prescribed by
            ``fmm_formulation``.
        inverse_z_permeability_matrix: The Fourier convolution matrix for the inverse
            of the z-component of the permeability.
        z_permeability_matrix: The Fourier convolution matrix for the z-component
            of the permeability.
        transverse_permeability_matrix: The transverse permittivity matrix.
        tangent_vector_field: The tangent vector field ``(tx, ty)`` used to compute the
            transverse permittivity matrix, if a vector FMM formulation is used. If
            the ``FFT`` formulation is used, the vector field is ``None``.
    """
    if formulation is Formulation.FFT:
        _transverse_permittivity_fn = functools.partial(
            fmm_matrices.transverse_permittivity_fft_anisotropic,
            expansion=expansion,
        )
        _transverse_permeability_fn = functools.partial(
            fmm_matrices.transverse_permeability_fft_anisotropic,
            expansion=expansion,
        )
        tangent_vector_field = None
    else:
        if isinstance(formulation, Formulation):
            vector_fn = vector.VECTOR_FIELD_SCHEMES[formulation.value]
        else:
            vector_fn = formulation
        tx, ty = vector_fn(vector_field_source, expansion, primitive_lattice_vectors)
        _transverse_permittivity_fn = functools.partial(
            fmm_matrices.transverse_permittivity_vector_anisotropic,
            tx=tx,
            ty=ty,
            expansion=expansion,
        )
        _transverse_permeability_fn = functools.partial(
            fmm_matrices.transverse_permeability_vector_anisotropic,
            tx=tx,
            ty=ty,
            expansion=expansion,
        )
        tangent_vector_field = (tx, ty)

    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    (
        permittivity_xx,
        permittivity_xy,
        permittivity_yx,
        permittivity_yy,
        permittivity_zz,
    ) = permittivities
    inverse_z_permittivity_matrix = _transform(1 / permittivity_zz)
    z_permittivity_matrix = _transform(permittivity_zz)
    transverse_permittivity_matrix = _transverse_permittivity_fn(
        permittivity_xx=permittivity_xx,
        permittivity_xy=permittivity_xy,
        permittivity_yx=permittivity_yx,
        permittivity_yy=permittivity_yy,
    )

    (
        permeability_xx,
        permeability_xy,
        permeability_yx,
        permeability_yy,
        permeability_zz,
    ) = permeabilities
    inverse_z_permeability_matrix = _transform(1 / permeability_zz)
    z_permeability_matrix = _transform(permeability_zz)
    transverse_permeability_matrix = _transverse_permeability_fn(
        permeability_xx=permeability_xx,
        permeability_xy=permeability_xy,
        permeability_yx=permeability_yx,
        permeability_yy=permeability_yy,
    )

    return (
        inverse_z_permittivity_matrix,
        z_permittivity_matrix,
        transverse_permittivity_matrix,
        inverse_z_permeability_matrix,
        z_permeability_matrix,
        transverse_permeability_matrix,
        tangent_vector_field,
    )


# -----------------------------------------------------------------------------
# Helper functions for validation and matrix assembly.
# -----------------------------------------------------------------------------


def _validate_and_broadcast(
    wavelength: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis.LatticeVectors,
    *permittivities: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, basis.LatticeVectors, Tuple[jnp.ndarray, ...]]:
    """Validates that shapes are compatible and adds required batch dimensions."""
    if not in_plane_wavevector.shape[-1] == 2:
        raise ValueError(
            f"`in_plane_wavevector` must have a final dimension of size 2 but got "
            f"a shape of {in_plane_wavevector.shape}."
        )

    if not all([permittivities[0].shape == p.shape for p in permittivities]):
        raise ValueError("Got permittivities with differing shapes.")

    permittivity = permittivities[0]
    if not misc.batch_compatible_shapes(
        wavelength.shape,
        in_plane_wavevector.shape[:-1],
        primitive_lattice_vectors.u.shape[:-1],
        primitive_lattice_vectors.v.shape[:-1],
        permittivity.shape[:-2],
    ):
        raise ValueError(
            f"`wavelength`, `in_plane_wavevector`, `primitive_lattice_vectors` "
            f"and `permittivity` must be batch-compatible, but got shapes of "
            f"{wavelength.shape}, {in_plane_wavevector.shape}, "
            f"{primitive_lattice_vectors.u.shape}, "
            f"{primitive_lattice_vectors.v.shape}, and {permittivity.shape}."
        )

    batch_shape = jnp.broadcast_shapes(
        wavelength.shape,
        in_plane_wavevector.shape[:-1],
        primitive_lattice_vectors.u.shape[:-1],
        primitive_lattice_vectors.v.shape[:-1],
        permittivity.shape[:-2],
    )
    wavelength = jnp.broadcast_to(wavelength, batch_shape)
    in_plane_wavevector = jnp.broadcast_to(in_plane_wavevector, batch_shape + (2,))
    primitive_lattice_vectors = basis.LatticeVectors(
        u=jnp.broadcast_to(primitive_lattice_vectors.u, batch_shape + (2,)),
        v=jnp.broadcast_to(primitive_lattice_vectors.v, batch_shape + (2,)),
    )

    permittivities = tuple(
        [jnp.broadcast_to(p, batch_shape + p.shape[-2:]) for p in permittivities]
    )
    return (
        wavelength,
        in_plane_wavevector,
        primitive_lattice_vectors,
        permittivities,
    )


def _select_eigenvalues_sign(eigenvalues: jnp.ndarray) -> jnp.ndarray:
    """Selects the sign of eigenvalues to have strictly positive imaginary part.

    Args:
        eigenvalues: The eigenvalues whose sign is to be adjusted.

    Returns:
        The eigenvalues with adjusted sign.
    """
    return jnp.where(jnp.imag(eigenvalues) < 0, -eigenvalues, eigenvalues)


# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


tree_util.register_pytree_node(
    Formulation,
    lambda x: ((), x.value),
    lambda value, _: Formulation(value),
)


tree_util.register_pytree_node(
    LayerSolveResult,
    lambda x: (
        (
            x.wavelength,
            x.in_plane_wavevector,
            x.primitive_lattice_vectors,
            x.expansion,
            x.eigenvalues,
            x.eigenvectors,
            x.z_permittivity_matrix,
            x.inverse_z_permittivity_matrix,
            x.z_permeability_matrix,
            x.inverse_z_permeability_matrix,
            x.transverse_permeability_matrix,
            x.tangent_vector_field,
        ),
        None,
    ),
    lambda _, x: LayerSolveResult(*x),
)
