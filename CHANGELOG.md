# Changelog
All notable changes to this project will be documented in this file.

<details><summary>0.4.0 - 4 May 2026</summary>

### Changed
- Modularized the extension into dedicated files:
	- `ld_helpers.py` for shared constants and helper/effect functions
	- `ld_node.py` for node class implementation
	- `__init__.py` as a lightweight entrypoint exporting node mappings

### Performance
- Reduced per-sigma overhead by precomputing conditioning conversion/encoding/tensor moves once per active resolution (base or hires), then reusing the prepared conditioning for each sigma evaluation.

</details>

<details><summary>0.3.1 - 3 May 2026</summary>

### Fixed
- Normalize outgoing `noise_mask` tensors to latent spatial resolution/batch size before returning from `LatentDetailer`, improving compatibility with stricter custom samplers
- Clamp reflect-blur radius to valid spatial bounds to avoid edge-case padding errors on very small latents.

</details>

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
