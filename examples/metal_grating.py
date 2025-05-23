"""A 1D metallic grating example, including convergence analysis.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import itertools
from typing import Tuple

import jax.numpy as jnp

import fmmax

NUM_TERMS_SWEEP = (9, 25, 49, 81, 121, 169, 225, 289, 361, 441, 529, 625, 729, 841)


def simulate_grating(
    permittivity_ambient: complex = 1.0 + 0.0j,
    permittivity_planarization: complex = 2.25 + 0.0j,
    permittivity_substrate: complex = -7.632 + 0.731j,
    wavelength_nm: float = 500.0,
    pitch_nm: float = 180.0,
    grating_width_nm: float = 60.0,
    grating_thickness_nm: float = 80.0,
    planarization_thickness_nm: float = 20.0,
    resolution_nm: float = 1.0,
    approximate_num_terms: int = 20,
    truncation: fmmax.Truncation = fmmax.Truncation.CIRCULAR,
    formulation: fmmax.Formulation = fmmax.Formulation.FFT,
) -> Tuple[int, complex, complex]:
    """Computes the TE- and TM-polarized reflection from a 1D stripe grating.

    Args:
        permittivity_ambient: The permittivity of the ambient.
        permittivity_planarization: The permittivity of media encapsulating grating.
        permittivity_substrate: The permittivity of the substrate below the grating,
            and the grating itself.
        wavelength_nm: The excitation wavelength, in nanometers.
        pitch_nm: The grating pitch, in nanometers.
        grating_width_nm: The width of the lines comprising the grating.
        grating_thickness_nm: The height of the grating.
        planarization_thickness_nm: The thickness of the planarization layer above
            the grating.
        resolution_nm: The rasterization resolution for patterned layers.
        approximate_num_terms: The approximate number of terms used in the plane
            wave expansion of the fields.
        truncation: Determines the truncation of the expansion.
        formulation: Specifies the formulation to be used.

    Returns:
        The number of terms in the expansion, and the reflection coefficients for TE-
        and TM-polarization.
    """
    x_nm, _ = jnp.meshgrid(
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        indexing="ij",
    )
    density = (jnp.abs(x_nm) <= grating_width_nm / 2).astype(float)

    permittivities = [
        jnp.asarray([[permittivity_ambient]]),
        jnp.asarray([[permittivity_planarization]]),
        fmmax.interpolate_permittivity(
            permittivity_solid=jnp.asarray(permittivity_substrate),
            permittivity_void=jnp.asarray(permittivity_planarization),
            density=density,
        ),
        jnp.asarray([[permittivity_substrate]]),
    ]
    thicknesses = [0, planarization_thickness_nm, grating_thickness_nm, 0]

    in_plane_wavevector = jnp.asarray([0.0, 0.0])
    primitive_lattice_vectors = fmmax.LatticeVectors(
        u=jnp.asarray([pitch_nm, 0.0]), v=jnp.asarray([0.0, pitch_nm])
    )
    expansion = fmmax.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=truncation,
    )
    layer_solve_results = [
        fmmax.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength_nm),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]
    s_matrix = fmmax.stack_s_matrix(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
    )

    r_te = s_matrix.s21[0, 0]
    r_tm = s_matrix.s21[expansion.num_terms, expansion.num_terms]
    return expansion.num_terms, complex(r_te), complex(r_tm)


def convergence_study(
    approximate_num_terms: Tuple[int, ...] = NUM_TERMS_SWEEP,
    truncations: Tuple[fmmax.Truncation, ...] = (
        fmmax.Truncation.CIRCULAR,
        fmmax.Truncation.PARALLELOGRAMIC,
    ),
    fmm_formulations: Tuple[fmmax.Formulation, ...] = (
        fmmax.Formulation.FFT,
        fmmax.Formulation.JONES_DIRECT,
        fmmax.Formulation.JONES,
        fmmax.Formulation.NORMAL,
        fmmax.Formulation.POL,
    ),
) -> Tuple[Tuple[fmmax.Formulation, fmmax.Truncation, int, complex, complex], ...]:
    """Sweeps over number of terms and fmm formulations to study convergence."""
    results = []
    for formulation, truncation, n in itertools.product(
        fmm_formulations,
        truncations,
        approximate_num_terms,
    ):
        num_terms, r_te, r_tm = simulate_grating(
            approximate_num_terms=n,
            truncation=truncation,
            formulation=formulation,
        )
        results.append((formulation, truncation, num_terms, r_te, r_tm))
        print(
            f"{formulation.value}/{truncation.value}/n={num_terms}: "
            f"r_te={complex(r_te):.3f}, r_tm={complex(r_tm):.3f}"
        )
    return tuple(results)


if __name__ == "__main__":
    convergence_study()
