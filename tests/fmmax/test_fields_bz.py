"""Tests for `fields` involving Brillouin zone integration.

Copyright (c) 2025 Martin F. Schubert
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fields, fmm, scattering, sources

jax.config.update("jax_enable_x64", True)


PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(u=basis.X, v=basis.Y)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=100,
    truncation=basis.Truncation.CIRCULAR,
)


class BZIntegratedFieldsTest(unittest.TestCase):
    @parameterized.expand([[0.314], [(0.314, 0.628)]])
    def test_fields_on_grid_match_expected(self, wavelength):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(3, 3),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        (
            _,
            _,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            _,
            _,
        ) = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )
        amplitudes_interior = fields.stack_amplitudes_interior_with_source(
            s_matrices_interior_before_source=((s_matrix, s_matrix),),
            s_matrices_interior_after_source=((s_matrix, s_matrix),),
            backward_amplitude_before_end=bwd_amplitude_before_end,
            forward_amplitude_after_start=fwd_amplitude_after_start,
        )

        # Manually carry out Brillouin zone integration.
        efield, hfield, _ = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            grid_shape=(30, 30),
            num_unit_cells=(3, 3),
        )
        efield_expected = [onp.mean(f, axis=(0, 1)) for f in efield]
        hfield_expected = [onp.mean(f, axis=(0, 1)) for f in hfield]

        # Automatically perform Brillouin zone integration.
        efield_integrated, hfield_integrated, _ = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            grid_shape=(30, 30),
            brillouin_grid_axes=(0, 1),
        )

        onp.testing.assert_allclose(efield_integrated, efield_expected)
        onp.testing.assert_allclose(hfield_integrated, hfield_expected)

    @parameterized.expand([[0.314], [(0.314, 0.628)]])
    def test_fields_on_coordinates_match_expected(self, wavelength):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(3, 3),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        (
            _,
            _,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            _,
            _,
        ) = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )
        amplitudes_interior = fields.stack_amplitudes_interior_with_source(
            s_matrices_interior_before_source=((s_matrix, s_matrix),),
            s_matrices_interior_after_source=((s_matrix, s_matrix),),
            backward_amplitude_before_end=bwd_amplitude_before_end,
            forward_amplitude_after_start=fwd_amplitude_after_start,
        )

        # Manually carry out Brillouin zone integration.
        x = jnp.arange(90) / 30
        y = jnp.zeros_like(x)
        efield, hfield, _ = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            x=x,
            y=y,
        )
        efield_expected = [onp.mean(f, axis=(0, 1)) for f in efield]
        hfield_expected = [onp.mean(f, axis=(0, 1)) for f in hfield]

        # Automatically perform Brillouin zone integration.
        efield_integrated, hfield_integrated, _ = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result, solve_result],
            layer_thicknesses=[thickness, thickness],
            layer_znum=(30, 30),
            x=x,
            y=y,
            brillouin_grid_axes=(0, 1),
        )

        onp.testing.assert_allclose(efield_integrated, efield_expected)
        onp.testing.assert_allclose(hfield_integrated, hfield_expected)

    @parameterized.expand([[(1, 1)], [(3, 3)]])
    def test_flux_integration(self, brillouin_grid_shape):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        bwd_amplitude_0_end, _, _, _, _, _ = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )

        _, flux = fields.amplitude_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_0_end),
            backward_amplitude=bwd_amplitude_0_end,
            layer_solve_result=solve_result,
        )
        expected_flux = onp.mean(flux, axis=(0, 1))  # Average over the BZ grid.
        expected_flux = onp.sum(expected_flux, axis=-2)  # Sum over Fourier orders.
        expected_flux /= onp.prod(brillouin_grid_shape)

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_0_end),
            backward_amplitude=bwd_amplitude_0_end,
            layer_solve_result=solve_result,
        )
        efield, hfield, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            brillouin_grid_axes=(0, 1),
        )
        flux_on_grid = fields.time_average_z_poynting_flux(efield, hfield)
        flux_on_grid = onp.mean(flux_on_grid, axis=(-3, -2))
        onp.testing.assert_allclose(flux_on_grid, expected_flux)

    @parameterized.expand([[(1, 1)], [(3, 3)]])
    def test_amplitudes_from_fields_from_amplitudes(self, brillouin_grid_shape):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        in_plane_wavevector = in_plane_wavevector[..., jnp.newaxis, :]
        solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.314),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.ones((1, 1)),
            expansion=EXPANSION,
        )
        thickness = jnp.asarray(1.0)
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result],
            layer_thicknesses=[thickness],
        )
        dipole = sources.gaussian_source(
            fwhm=0.1,
            location=jnp.asarray([[1.5, 1.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        bwd_amplitude_0_end, _, _, _, _, _ = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=jnp.zeros_like(dipole),
            jz=dipole,
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_0_end),
            backward_amplitude=bwd_amplitude_0_end,
            layer_solve_result=solve_result,
        )
        efield, hfield, _ = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=solve_result,
            shape=(100, 100),
            brillouin_grid_axes=(0, 1),
        )

        fwd, bwd = sources.amplitudes_for_fields(
            ex=efield[0],
            ey=efield[1],
            hx=hfield[0],
            hy=hfield[1],
            layer_solve_result=solve_result,
            brillouin_grid_axes=(0, 1),
        )

        onp.testing.assert_allclose(bwd, bwd_amplitude_0_end, atol=1e-12)
