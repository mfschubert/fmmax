"""Tests for `fmmax.basis`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
from jax import tree_util
from parameterized import parameterized

from fmmax import basis


def _coeffs_set(coeffs):
    return {tuple([int(idx) for idx in c]) for c in coeffs}


class ExpansionTest(unittest.TestCase):
    def test_circular_with_square_lattice(self):
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.array([1.0, 0.0]), v=jnp.array([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=20,
            truncation=basis.Truncation.CIRCULAR,
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.u, onp.array([1.0, 0.0])
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.v, onp.array([0.0, 1.0])
        )
        # Circular truncation discards the corner elements.
        circular_mask = jnp.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ]
        ).astype(bool)
        i, j = jnp.meshgrid(jnp.arange(-2, 3), jnp.arange(-2, 3), indexing="ij")
        expected_coefficients = jnp.stack(
            [i[circular_mask].flatten(), j[circular_mask].flatten()], axis=-1
        )
        self.assertSequenceEqual(
            _coeffs_set(expansion.basis_coefficients),
            _coeffs_set(expected_coefficients),
        )

    def test_circular_with_rectangle_lattice(self):
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.array([2.0, 0.0]), v=jnp.array([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=20,
            truncation=basis.Truncation.CIRCULAR,
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.u, onp.array([0.5, 0.0])
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.v, onp.array([0.0, 1.0])
        )
        # Circular truncation discards the corner elements.
        circular_mask = jnp.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 0],
            ]
        ).astype(bool)
        i, j = jnp.meshgrid(jnp.arange(-3, 4), jnp.arange(-1, 2), indexing="ij")
        expected_coefficients = jnp.stack(
            [i[circular_mask].flatten(), j[circular_mask].flatten()], axis=-1
        )
        self.assertSequenceEqual(
            _coeffs_set(expansion.basis_coefficients),
            _coeffs_set(expected_coefficients),
        )

    def test_parallelogramic_with_square_lattice(self):
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.array([1.0, 0.0]), v=jnp.array([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=20,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.u, onp.array([1.0, 0.0])
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.v, onp.array([0.0, 1.0])
        )
        # Parallelogramic truncation includes all coefficients in the range `(-2, +2)`.
        i, j = jnp.meshgrid(jnp.arange(-2, 3), jnp.arange(-2, 3), indexing="ij")
        expected_coefficients = jnp.stack([i.flatten(), j.flatten()], axis=-1)
        self.assertSequenceEqual(
            _coeffs_set(expansion.basis_coefficients),
            _coeffs_set(expected_coefficients),
        )

    def test_parallelogramic_with_rectangle_lattice(self):
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.array([2.0, 0.0]), v=jnp.array([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=25,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.u, onp.array([0.5, 0.0])
        )
        onp.testing.assert_allclose(
            primitive_lattice_vectors.reciprocal.v, onp.array([0.0, 1.0])
        )
        i, j = jnp.meshgrid(jnp.arange(-3, 4), jnp.arange(-1, 2), indexing="ij")
        expected_coefficients = jnp.stack([i.flatten(), j.flatten()], axis=-1)
        self.assertSequenceEqual(
            _coeffs_set(expansion.basis_coefficients),
            _coeffs_set(expected_coefficients),
        )

    def test_expansion_treedef(self):
        # Checks that equality checks for the treedef of an `Expansion` are possible.
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.array([2.0, 0.0]), v=jnp.array([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=20,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        treedef = tree_util.tree_structure(expansion)
        self.assertEqual(treedef, treedef)

    def test_expansion_flatten_unflatten(self):
        # Checks that equality checks for the treedef of an `Expansion` are possible.
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.array([2.0, 0.0]), v=jnp.array([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=20,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        leaves, treedef = tree_util.tree_flatten(expansion)
        restored = tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(treedef, tree_util.tree_structure(restored))
        for a, b in zip(leaves, tree_util.tree_leaves(restored)):
            onp.testing.assert_array_equal(a, b)

    @parameterized.expand(
        [
            [(1, 0), (0, 1), basis.Truncation.CIRCULAR],
            [(1, 0), (0, 1), basis.Truncation.PARALLELOGRAMIC],
            [(2, 0), (0, 1), basis.Truncation.CIRCULAR],
            [(2, 0), (0, 1), basis.Truncation.PARALLELOGRAMIC],
            [(1, 1), (1, -1), basis.Truncation.CIRCULAR],
            [(1, 1), (1, -1), basis.Truncation.PARALLELOGRAMIC],
        ]
    )
    def test_coefficient_ordering_independent_of_num_terms(self, u, v, truncation):
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.asarray(u), v=jnp.asarray(v)
        )
        expansion_20 = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=20,
            truncation=truncation,
        )
        expansion_50 = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=truncation,
        )
        expansion_100 = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=truncation,
        )
        coeffs_20 = expansion_20.basis_coefficients
        coeffs_50 = expansion_50.basis_coefficients
        coeffs_100 = expansion_100.basis_coefficients
        onp.testing.assert_array_equal(coeffs_20, coeffs_50[: coeffs_20.shape[0], :])
        onp.testing.assert_array_equal(coeffs_20, coeffs_100[: coeffs_20.shape[0], :])
        onp.testing.assert_array_equal(coeffs_50, coeffs_100[: coeffs_50.shape[0], :])


class InPlaneWavevectorTest(unittest.TestCase):
    @parameterized.expand([[(1,)], [(1, 2, 3)], [(0, 1)]])
    def test_brillouin_grid_shape_validation(self, invalid_shape):
        with self.assertRaisesRegex(
            ValueError, "`brillouin_grid_shape` must be length-2 with"
        ):
            basis.brillouin_zone_in_plane_wavevector(
                brillouin_grid_shape=invalid_shape,
                primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y),
            )

    @parameterized.expand(
        [
            [(1, 1), [[[0, 0]]]],
            [(2, 1), [[[-0.5 * jnp.pi, 0]], [[0, 0]]]],
            [(1, 2), [[[0, -jnp.pi], [0, 0]]]],
            [(3, 1), [[[-jnp.pi / 3, 0]], [[0, 0]], [[jnp.pi / 3, 0]]]],
            [(1, 3), [[[0, -2 * jnp.pi / 3], [0, 0], [0, 2 * jnp.pi / 3]]]],
        ],
    )
    def test_brillouin_wavevector_matches_exected(self, shape, expected_vectors):
        primitive_lattice_vectors = basis.LatticeVectors(basis.X * 2, basis.Y)
        wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=shape,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        onp.testing.assert_allclose(wavevector, expected_vectors)

    @parameterized.expand(
        [[(1, 1)], [(3, 3)], [(5, 5)]],
    )
    def test_brillouin_wavevector_at_center_is_exactly_zero(self, shape):
        primitive_lattice_vectors = basis.LatticeVectors(basis.X * 2, basis.Y)
        wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=shape,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        onp.testing.assert_array_equal(
            wavevector[shape[0] // 2, shape[1] // 2, :],
            onp.asarray([0, 0]),
        )

    @parameterized.expand(
        [
            [1.0, 0.0, 0.0, 1.0, (0, 0)],
            [1.0, jnp.pi / 4, 0.0, 1.0, (2 * jnp.pi / jnp.sqrt(2), 0)],
            [1.0, jnp.pi / 4, jnp.pi / 2, 1.0, (0, 2 * jnp.pi / jnp.sqrt(2))],
        ]
    )
    def test_plane_wave_wavevector_matches_expected(
        self, wavelength, polar_angle, azimuthal_angle, permittivity, expected
    ):
        wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=wavelength,
            polar_angle=polar_angle,
            azimuthal_angle=azimuthal_angle,
            permittivity=permittivity,
        )
        onp.testing.assert_allclose(wavevector, expected, rtol=1e-6, atol=1e-6)


class UnitCellCoordiantesTest(unittest.TestCase):
    @parameterized.expand(
        [
            [(1, 1), (3, 2), (0.5, 1.5, 2.5), (0.5, 1.5)],
            [(1, 1), (4, 3), (0.5, 1.5, 2.5, 3.5), (0.5, 1.5, 2.5)],
            [(2, 2), (2, 1), (0.25, 0.75, 1.25, 1.75), (0.25, 0.75)],
        ]
    )
    def test_values_match_expected(self, shape, num_unit_cells, expected_x, expected_y):
        x, y = basis.unit_cell_coordinates(
            primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
            shape=shape,
            num_unit_cells=num_unit_cells,
        )
        onp.testing.assert_array_equal(
            x, jnp.broadcast_to(jnp.asarray(expected_x)[:, jnp.newaxis], x.shape)
        )
        onp.testing.assert_array_equal(
            y, jnp.broadcast_to(jnp.asarray(expected_y)[jnp.newaxis, :], y.shape)
        )


class GenerateBasisTest(unittest.TestCase):
    def test_parallelogramic_num_terms_is_monotonic(self):
        approximate_num_terms = onp.arange(50, 250)
        primitive_lattice_vectors = basis.LatticeVectors(
            u=1.5 * basis.X,
            v=0.75 * basis.Y,
        )
        num_terms = []
        for n in approximate_num_terms:
            coeffs = basis._basis_coefficients_parallelogramic(
                primitive_lattice_vectors,
                approximate_num_terms=n,
            )
            num_terms.append(coeffs.shape[0])
        for n, n_next in zip(num_terms[:-1], num_terms[1:]):
            self.assertGreaterEqual(n_next, n)

    def test_circular_num_terms_is_monotonic(self):
        approximate_num_terms = onp.arange(50, 250)
        primitive_lattice_vectors = basis.LatticeVectors(
            u=1.5 * basis.X,
            v=0.75 * basis.Y,
        )
        num_terms = []
        for n in approximate_num_terms:
            coeffs = basis._basis_coefficients_circular(
                primitive_lattice_vectors,
                approximate_num_terms=n,
            )
            num_terms.append(coeffs.shape[0])
        for n, n_next in zip(num_terms[:-1], num_terms[1:]):
            self.assertGreaterEqual(n_next, n)


class ShapeValidationTest(unittest.TestCase):
    @parameterized.expand(
        [
            (onp.asarray([[-2, -2], [2, 2]]), (5, 5)),
            (onp.asarray([[-5, -2], [5, 2]]), (11, 5)),
            (onp.asarray([[0, -2], [8, 3]]), (17, 7)),
        ]
    )
    def test_min_shape(self, basis_coefficients, expected_min_shape):
        expansion = basis.Expansion(basis_coefficients=basis_coefficients)
        min_shape = basis.min_array_shape_for_expansion(expansion)
        self.assertSequenceEqual(min_shape, expected_min_shape)

    def test_shape_validation(self):
        expansion = basis.Expansion(basis_coefficients=jnp.asarray([[-2, -2], [2, 2]]))
        with self.assertRaisesRegex(ValueError, "`shape` is insufficient for"):
            basis.validate_shape_for_expansion((8, 4), expansion)
