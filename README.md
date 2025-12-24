# ComfyUI-SpectralVAEDetailer

<p align="center">
<img src="https://github.com/user-attachments/assets/2ece32c1-fa64-46ec-9921-8181b200d6dd" width=500>
</p>

A node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that adjusts a latent image before the VAE decoding step in order to improve your image quality.

By default, the Spectral Detailer will:

- Tame harsh, unnatural highlights and shadows.
- Inject micro-grain (band-pass, VAE-friendly noise) that reads as photographic texture instead of “AI smoothness”, while avoiding big shifts in overall lighting.
- Improve perceived micro-detail by projecting a single, low-cost UNet “denoised” estimate back into the latent and selectively reinjecting high-frequency structure.
- Provide an optional, dedicated CFG-delta injection path so CFG can influence the final latent adjustment in a controlled way.

View [related thread on Reddit](https://reddit.com/r/StableDiffusion/comments/1pucj4g/spectral_vae_detailer_new_way_to_squeeze_out_more/) for demo images.

It was specifically designed to boost photorealism in SDXL. Demo images generated with Snakebite 2.4 Turbo. If you're using a different checkpoint or VAE, you may need to tweak the parameters for best results.

---

## How it works (high level)

SpectralVAEDetailer performs **one extra UNet evaluation** at a chosen `sigma`, then:

1. Computes a denoised latent estimate.
2. Splits that delta into low-frequency vs high-frequency components using a fast blur (`avg_pool2d`).
3. Reinforces the high-frequency component with safety gating (`protect_lows`) to avoid wrecking smooth gradients.
4. Optionally adds a **separate CFG-delta injection** (high/low split) so guidance has an observable effect without overpowering the detail pass.
5. Adds **band-pass correlated micrograin** (`noise_radius`) that is stable and VAE-friendly (and can be suppressed in flat regions).

Everything happens in latent space, before VAE decode.

## Installation

1. Go to your ComfyUI `custom_nodes` directory:
   - `ComfyUI/custom_nodes/`

2. Clone this repository:
   - `git clone https://github.com/YOURNAME/ComfyUI-SpectralVAEDetailer.git`

3. Restart ComfyUI.

Find the node in:

**Add Node → latent → postprocess → SpectralVAEDetailer**

This node is also available on the Comfy Registry.

## Usage

Place **SpectralVAEDetailer** after your sampler (when you already have a `LATENT`) and before **VAE Decode**.

Typical chain:

`KSampler → SpectralVAEDetailer → VAE Decode → Preview Image`

## Key parameters

### `sigma`

The noise level used for the single UNet evaluation. Lower values tend to be subtler; higher values can flatten contrast more aggressively.

### `detail_strength`, `hf_radius`, `protect_lows`

Control how much high-frequency structure is reinjected and how strongly smooth gradients are protected.

### `cfg`, `cfg_hf_boost`, `cfg_radius`

Optional guidance-delta injection. If you want CFG to “show up” in the detailer stage, increase `cfg_hf_boost`.

### `noise_scale`, `noise_radius`, `noise_flat_suppress`

Latent micrograin injection:

- `noise_radius`: controls grain size / cluster scale (band-pass).
- `noise_flat_suppress`: suppress grain in flat areas (`0` = grain everywhere).

## Tips

- If your image looks too flat:
  - reduce `hf_radius` or `mid_strength`, or lower `sigma`.

- If you see grain in skies / flat backgrounds:
  - raise `noise_scale`.

- If your images look blurry:
  - reduce `radius` and `chroma` parameters.

## Notes

This node is intentionally *speed-oriented*: it uses one extra UNet pass plus lightweight tensor ops.

It's primarily tuned for SDXL photorealism workflows, but can be experimented with on other latent diffusion models.
