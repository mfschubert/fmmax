"""Tests for `slant` module.

Copyright (c) 2024 Martin F. Schubert
"""

import functools
import unittest

import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fmm, scattering, translate


def xy(primitive_lattice_vectors, shape):
    i, j = jnp.meshgrid(
        jnp.arange(shape[0]) / shape[0],
        jnp.arange(shape[1]) / shape[1],
        indexing="ij",
    )
    x = i * primitive_lattice_vectors.u[0] + j * primitive_lattice_vectors.v[0]
    y = i * primitive_lattice_vectors.u[1] + j * primitive_lattice_vectors.v[1]
    return x, y


def circular_permittivity(primitive_lattice_vectors, dx, dy, shape):
    x, y = xy(primitive_lattice_vectors, shape)
    x0 = (jnp.amax(x) + jnp.amin(x)) / 2
    y0 = (jnp.amax(y) + jnp.amin(y)) / 2
    r = jnp.sqrt((x - x0 - dx) ** 2 + (y - y0 - dy) ** 2)
    return 1 + 1 * jnp.exp(-(r**2) * 40)


class TranslateLayerSolveResultTest(unittest.TestCase):
    @parameterized.expand(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.2, 0.0],
            [1.0, 1.0, 0.2, 0.1],
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.1],
            [1.0, 2.0, 0.2, 0.0],
            [1.0, 2.0, 0.2, 0.1],
        ]
    )
    def test_matrices(self, period_x, period_y, dx, dy):
        # Compare matrices from manually-shifted permittivity distributions to those
        # obatained by simply translating the eigensolve
        primitive_lattice_vectors = basis.LatticeVectors(
            u=period_x * basis.X,
            v=period_y * basis.Y,
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )
        permittivity = circular_permittivity(
            primitive_lattice_vectors=primitive_lattice_vectors,
            dx=0,
            dy=0,
            shape=(100, 100),
        )
        permittivity_shifted = jnp.roll(
            permittivity,
            (int(100 * dx / period_x), int(100 * dy / period_y)),
            axis=(0, 1),
        )
        eigensolve = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=jnp.asarray(0.55),
            in_plane_wavevector=jnp.zeros((2,)),
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        solve_result = eigensolve(permittivity=permittivity)
        solve_result_shifted = translate.translate_layer_solve_result(
            solve_result, dx=jnp.asarray(dx), dy=jnp.asarray(dy)
        )
        solve_result_expected = eigensolve(permittivity=permittivity_shifted)

        with self.subTest("z_permittivity_matrix"):
            onp.testing.assert_allclose(
                solve_result_shifted.z_permittivity_matrix,
                solve_result_expected.z_permittivity_matrix,
                atol=1e-7,
            )
        with self.subTest("z_permeability_matrix"):
            onp.testing.assert_allclose(
                solve_result_shifted.z_permeability_matrix,
                solve_result_expected.z_permeability_matrix,
                atol=1e-7,
            )
        with self.subTest("inverse_z_permittivity_matrix"):
            onp.testing.assert_allclose(
                solve_result_shifted.inverse_z_permittivity_matrix,
                solve_result_expected.inverse_z_permittivity_matrix,
                atol=1e-7,
            )
        with self.subTest("inverse_z_permeability_matrix"):
            onp.testing.assert_allclose(
                solve_result_shifted.inverse_z_permeability_matrix,
                solve_result_expected.inverse_z_permeability_matrix,
                atol=1e-7,
            )
        with self.subTest("transverse_permeability_matrix"):
            onp.testing.assert_allclose(
                solve_result_shifted.transverse_permeability_matrix,
                solve_result_expected.transverse_permeability_matrix,
                atol=1e-7,
            )
        with self.subTest("omega_script_k_matrix"):
            onp.testing.assert_allclose(
                solve_result_shifted.omega_script_k_matrix,
                solve_result_expected.omega_script_k_matrix,
                atol=1e-4,  # Looser tolerance here.
            )

    @parameterized.expand(
        [
            [(1, 0), (0, 1), 0.0, 0.0],
            [(1, 0), (0, 1), 0.2, 0.0],
            [(1, 0), (0, 1), 0.0, 0.2],
            [(1, 0.1), (0.2, 1), 0.2, 0.0],
            [(1, 0.1), (0.1, 1), 0.0, 0.2],
        ]
    )
    def test_smatrix_is_translation_invariant(self, u, v, dx, dy):
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.asarray(u),
            v=jnp.asarray(v),
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )

        eigensolve = functools.partial(
            fmm.eigensolve_isotropic_media,
            wavelength=jnp.asarray(0.55),
            in_plane_wavevector=jnp.zeros((2,)),
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

        solve_result_ambient = eigensolve(permittivity=jnp.ones((1, 1)))
        permittivity = circular_permittivity(
            primitive_lattice_vectors=primitive_lattice_vectors,
            dx=0,
            dy=0,
            shape=(100, 100),
        )
        solve_result_circle = eigensolve(permittivity=permittivity)
        solve_result_circle_shifted = translate.translate_layer_solve_result(
            solve_result_circle,
            dx=jnp.asarray(dx),
            dy=jnp.asarray(dy),
        )
        thicknesses = [jnp.zeros(()), jnp.ones(()), jnp.zeros(())]

        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[
                solve_result_ambient,
                solve_result_circle,
                solve_result_ambient,
            ],
            layer_thicknesses=thicknesses,
        )
        s_matrix_shifted = scattering.stack_s_matrix(
            layer_solve_results=[
                solve_result_ambient,
                solve_result_circle_shifted,
                solve_result_ambient,
            ],
            layer_thicknesses=thicknesses,
        )

        # Directly compare zeroth-order transmission.
        with self.subTest("s11"):
            onp.testing.assert_allclose(
                s_matrix_shifted.s11[0, 0], s_matrix.s11[0, 0], atol=1e-4
            )
        with self.subTest("s12"):
            onp.testing.assert_allclose(
                s_matrix_shifted.s12[0, 0], s_matrix.s12[0, 0], atol=1e-4
            )
        with self.subTest("s21"):
            onp.testing.assert_allclose(
                s_matrix_shifted.s21[0, 0], s_matrix.s21[0, 0], atol=1e-4
            )
        with self.subTest("s22"):
            onp.testing.assert_allclose(
                s_matrix_shifted.s22[0, 0], s_matrix.s22[0, 0], atol=1e-4
            )
