# ComfyUI-LatentDetailer

<p align="center">
<img src="https://github.com/user-attachments/assets/2ece32c1-fa64-46ec-9921-8181b200d6dd" width=500>
</p>

 *(Formerly ComfyUI-SpectralVAEDetailer)*

A node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that adjusts a latent image before the VAE decoding step in order to improve your image quality.

It performs a fast latent-space “detail pass”: runs one (or more) extra UNet evaluation(s) at configurable sigma positions, then reinjects controlled high-frequency structure + optional CFG delta, with lightweight post-ops (micrograin, bloom/halation, color drift, chroma controls).

Also published on the **Comfy Registry** (search for “LatentDetailer”).

View the [related thread on Reddit](https://reddit.com/r/StableDiffusion/comments/1pucj4g/spectral_vae_detailer_new_way_to_squeeze_out_more/) for demo images.

> [!TIP]
> If the effect is too strong, please check the `example_workflows` folder for recommended presets!

## What it does (high level)

1. Computes a denoised estimate from the current latent using UNet at one or more sigma positions (`sigmas`).
2. Extracts / boosts high-frequency structure and reinjects it into the latent (with optional “protect lows” logic).
3. Optionally injects a guidance-delta term (CFG delta) to ensure guidance has a visible impact in this stage.
4. Optional finishing:
   - micrograin injection
   - bloom / halation approximation (luma-only)
   - color drift
   - chromatic aberration (spatial chroma misregistration)

## Key inputs

### `sigmas` (STRING)
Comma-delimited list of normalized sigma positions in **[0..1]**.  
Each entry triggers one UNet eval; results are averaged.

- `0.0` = early/noisy (largest sigma)
- `1.0` = late/clean (smallest sigma)

Examples:
- `0.40` (default)
- `0.25,0.55`

### Hi-res fix
- `hires_scale` (FLOAT, 1.0–4.0): fractional allowed (e.g. 1.5)
- `hires_strength`: how much of the hi-res residual is applied back at 1x
- `hires_use_importance_mask` + `hires_mask_strength`: focuses work where the base latent has more structure

### Detail / structure
- `detail_strength`, `hf_radius`, `protect_lows`: primary detail reinjection controls
- `mid_strength`: optional mid-frequency shaping

### Chroma controls
- `detail_chroma`: scales how strongly detail + CFG injections affect **chroma latent channels** (1..3).
- `chromatic_aberration`: lens-like spatial misregistration applied to chroma channels (1..3) **after** all other adjustments. Keep subtle (typically ~0.02–0.10).

### Bloom / halation
- `bloom_strength`: spreads bright luma highlights (channel 0) into a soft glow
- `bloom_threshold`: highlight threshold
- `bloom_radius`: blur radius (bigger = softer, slower)

### Grain
- `noise_scale`, `noise_radius`, `noise_flat_suppress`: micrograin injection controls
- `grain_luma`: modulates grain by luma (more in shadows, less in highlights)


## Notes

This node is intentionally speed-oriented: by default, it adds **only one** extra UNet passes plus lightweight tensor ops.

It’s primarily tuned for SDXL photoreal workflows, but can be experimented with on other latent diffusion models.
