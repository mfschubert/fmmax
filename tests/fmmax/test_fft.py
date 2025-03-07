"""Tests for `fmmax.fft`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp

from fmmax import _fft, basis


PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(
    u=jnp.array([1, 0]), v=jnp.array([0, 1])
)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=30,
    truncation=basis.Truncation.CIRCULAR,
)


class FftTest(unittest.TestCase):
    def testfft_ifft_single(self):
        shape = (100, 80)
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=shape)

        y = _fft.fft(x, EXPANSION)
        self.assertSequenceEqual(y.shape, (EXPANSION.num_terms,))

        ifft_y = _fft.ifft(y, EXPANSION, shape)
        self.assertSequenceEqual(ifft_y.shape, shape)

        fft_ifft_y = _fft.fft(ifft_y, EXPANSION)
        onp.testing.assert_allclose(y, fft_ifft_y)

    def testfft_specify_axis(self):
        shape = (20, 100, 80, 4)
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=shape)
        y = _fft.fft(x, EXPANSION, axes=(-3, -2))
        expected_y_transpose = _fft.fft(
            jnp.transpose(x, (0, 3, 1, 2)),
            EXPANSION,
            axes=(-2, -1),
        )
        expected_y = jnp.transpose(expected_y_transpose, (0, 2, 1))
        onp.testing.assert_array_equal(y, expected_y)

    def test_ifft_specify_axis(self):
        shape = (20, 100, 80)
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=shape)
        y = _fft.fft(x, EXPANSION)
        expected_ifft_y = _fft.ifft(y, EXPANSION, shape=(100, 80))
        self.assertSequenceEqual(y.shape, (20, EXPANSION.num_terms))

        y_transposed = jnp.transpose(y, axes=(1, 0))
        ifft_y_transposed = _fft.ifft(y_transposed, EXPANSION, shape=(100, 80), axis=-2)
        ifft_y = jnp.transpose(ifft_y_transposed, axes=(2, 0, 1))
        onp.testing.assert_allclose(ifft_y, expected_ifft_y)

    def testfft_ifft_batch(self):
        batch_size = 24
        shape = (100, 80)
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch_size,) + shape)

        y = _fft.fft(x, EXPANSION)
        self.assertSequenceEqual(y.shape, (batch_size, EXPANSION.num_terms))

        ifft_y = _fft.ifft(y, EXPANSION, shape)
        self.assertSequenceEqual(ifft_y.shape, (batch_size,) + shape)

        fft_ifft_y = _fft.fft(ifft_y, EXPANSION)
        onp.testing.assert_allclose(y, fft_ifft_y)


class ToeplitzIndicesTest(unittest.TestCase):
    def test_standard(self):
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            approximate_num_terms=30,
            truncation=basis.Truncation.CIRCULAR,
        )
        indices = _fft._standard_toeplitz_indices(expansion)
        for row_coeff, row in zip(expansion.basis_coefficients, indices):
            for col_coeff, idx in zip(expansion.basis_coefficients, row):
                self.assertSequenceEqual(row_coeff.shape, (2,))
                self.assertSequenceEqual(col_coeff.shape, (2,))
                self.assertSequenceEqual(idx.shape, (2,))
                onp.testing.assert_array_equal(-idx + row_coeff, col_coeff)


class ShapeValidationTest(unittest.TestCase):
    def test_fourier_convolution_matrix(self):
        expansion = basis.Expansion(basis_coefficients=jnp.asarray([[-2, -2], [2, 2]]))
        with self.assertRaisesRegex(ValueError, "`shape` is insufficient for"):
            _fft.fourier_convolution_matrix(jnp.ones((8, 4)), expansion)

    def testfft(self):
        expansion = basis.Expansion(basis_coefficients=jnp.asarray([[-2, -2], [2, 2]]))
        with self.assertRaisesRegex(ValueError, "`shape` is insufficient for"):
            _fft.fft(jnp.ones((8, 4)), expansion)

    def test_ifft(self):
        expansion = basis.Expansion(basis_coefficients=jnp.asarray([[-2, -2], [2, 2]]))
        with self.assertRaisesRegex(ValueError, "`shape` is insufficient for"):
            _fft.ifft(jnp.ones((2,)), expansion, shape=(8, 4))
