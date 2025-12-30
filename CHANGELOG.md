# Changelog
All notable changes to this project will be documented in this file.

<details><summary>0.3.0 - 28 December 2025</summary>

### Added
- Bloom controls (`bloom_strength`, `bloom_threshold`, `bloom_radius`)
- `chromatic_abberation`
- `grain_luma`
- `noise_tail`

### Changed
- Node renamed from `ComfyUI-SpectralVAEDetailer` to `ComfyUI-LatentDetailer`
- The `hires_scale` parameter is now a float instead of integer
- Renamed `debug_print` to `debug`
- Lowered default `noise_scale` from 0.2 to 0.1

### Fixed
- Reflect padding error when the radius is >= the latent spatial dimension

</details>

<details><summary>0.2.0 - 23 December 2025</summary>

### Added
- Input `adaptive_cfg_radius`
- Multiple soft clipping inputs

</details>

<details><summary>0.0.1 - 23 December 2025</summary>

### Added
- Initial release

</details>