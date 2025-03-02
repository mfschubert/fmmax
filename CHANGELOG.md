# Change log

## Unreleased
- Set `Formulation.JONES_DIRECT_FOURIER` as the default formulation for all eigensolve functions in `fmm` module.

## 0.14.4 (February 25, 2025)
- Set `Formulation.JONES_DIRECT_FOURIER` as the default formulation for `fmm.eigensolve_isotropic_media`.
- Set `Truncation.CIRCULAR` as the default truncation for `basis.generate_expansion`.
- Expand documentation for `fmm.Formulation` enum, describing each formulation.

## 0.13.4 (January 30, 2025)
- Add a `fields.time_average_z_poynting_flux` function to compute the real-space z-oriented Poynting flux from the real-space electromagnetic fields.

## 0.13.3 (January 25, 2025)
- Avoid post-init validation for `fmm.LayerSolveResult` when the attributes are not arrays, e.g. tracer objects. This allows the `LayerSolveResult` to be returned by a jit-ed function.

## 0.13.2 (January 24, 2025)
- Adjust shapes of transverse and z permeability matrices for uniform isotropic media to ensure they have shapes matching those obtained when performing a patterned layer eigensolve. This allows solve reuslts to be concatenated.

## 0.13.1 (January 15, 2025)
- Relax minimum jax dependency to `>= 0.4.27`. There is a bug in some newer jax versions which causes hanging with the CPU backend (https://github.com/jax-ml/jax/issues/24219).

## 0.13.0 (January 14, 2025)
- Reorganize the project to make the `_fmm_matrices` and `_fft` modules are made private.
- Update the build CI workflow to test older, known-good jax versions.
