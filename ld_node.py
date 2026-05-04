from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .ld_helpers import *

class SpectralVAEDetailer:

	def __init__(self):
		self._conv_cache = {}
		self._enc_cache = {}
		self._grid_cache = {}

	@classmethod
	def INPUT_TYPES(cls):
		return {
		    "required": {
		        # --- Top controls
		        "seed": ("INT", {
		            "default": -1,
		            "min": -1,
		            "max": 2**31 - 1,
		            "step": 1,
		            "tooltip": "Random seed for grain/color drift. Use -1 for random each run."
		        }),
		        "sigmas": ("STRING", {
		            "default": "0.40",
		            "multiline": False,
		            "tooltip": "Comma-delimited list of normalized sigma positions (0..1). Each entry runs one UNet evaluation and the denoised estimates are averaged. Example: 0.25,0.55. 0=start (noisiest), 1=end (cleanest)."
		        }),

		        # --- Main inputs
		        "model": ("MODEL", ),
		        "latent": ("LATENT", ),
		        "positive": ("CONDITIONING", ),
		        "negative": ("CONDITIONING", ),

		        # --- Uniformity fixes
		        "luma_clarity": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Mid-band local contrast on latent channel 0. 1.0 is intentionally strong."
		        }),
		        "boost_confidence": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Boosts UNet-proposed micro detail ONLY where it appears confident; suppresses flats and strong edges."
		        }),
		        # --- Bloom (approx photographic halation / highlight spread)
		        "bloom_strength": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Approximate photographic bloom/halation by spreading bright highlights (luma only). Best paired with a little noise_scale (e.g. 0.02–0.08) for a more convincing photographic rolloff."
		        }),
		        "bloom_threshold": ("FLOAT", {
		            "default": 0.65,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Only highlights above this luma threshold contribute to bloom."
		        }),
		        "bloom_radius": ("INT", {
		            "default": 8,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "Blur radius for bloom spread. Larger is softer but slower."
		        }),

		        # --- Color drift
		        "color_drift": ("FLOAT", {
		            "default": 0.25,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Noise-driven granular color drift (micro color distribution). Higher = stronger."
		        }),
		        "color_drift_radius": ("INT", {
		            "default": 16,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "Granular drift scale. 1..3 is typical; 16 is very broad/slow drift."
		        }),

		        # --- CFG group
		        "cfg": ("FLOAT", {
		            "default": 7.0,
		            "min": 0.0,
		            "max": 10.0,
		            "step": 0.05,
		            "tooltip": "Base CFG. This node applies additional HF/LF shaping when cfg > 1."
		        }),
		        "cfg_hf_boost": ("FLOAT", {
		            "default": 5.0,
		            "min": 0.0,
		            "max": 5.0,
		            "step": 0.05,
		            "tooltip": "How strongly to inject high-frequency CFG detail (from den_pos - den_neg)."
		        }),
		        "cfg_lf_boost": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.02,
		            "tooltip": "How strongly to inject low-frequency CFG contrast (usually keep low)."
		        }),
		        "cfg_radius": ("INT", {
		            "default": 5,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "CFG split radius. In adaptive mode, this is the DETAIL radius."
		        }),
		        "cfg_radius_flat": ("INT", {
		            "default": 0,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "CFG split radius used in flat/low-detail regions when adaptive mode is ON."
		        }),
		        "cfg_radius_adaptive": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "If ON, blends between cfg_radius_flat (flat) and cfg_radius (detail)."
		        }),
		        "cfg_adapt_feather": ("INT", {
		            "default": 2,
		            "min": 0,
		            "max": 32,
		            "step": 1,
		            "tooltip": "Blur radius applied to the adaptive mask. Higher reduces halos but can soften detail reach."
		        }),
		        "cfg_adapt_gamma": ("FLOAT", {
		            "default": 2.0,
		            "min": 0.5,
		            "max": 3.0,
		            "step": 0.05,
		            "tooltip": "Mask curve. >1 shrinks 'detailed' regions (less spill/halo). <1 expands them."
		        }),

		        # --- Core look
		        "detail_strength": ("FLOAT", {
		            "default": 0.65,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.01,
		            "tooltip": "Strength of injected high-frequency detail from denoised estimate."
		        }),
		        "hf_radius": ("INT", {
		            "default": 4,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "Detail split radius for base projection (larger = coarser separation)."
		        }),
		        "mid_strength": ("FLOAT", {
		            "default": 0.05,
		            "min": 0.0,
		            "max": 0.5,
		            "step": 0.01,
		            "tooltip": "Adds some mid/low component of the base projection (contrast/shape)."
		        }),
		        "detail_chroma": ("FLOAT", {
		            "default": 0.1,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.01,
		            "tooltip": "Scales how much the detail + CFG injections affect chroma latent channels (1..3). This is NOT chromatic aberration (no spatial shift)."
		        }),
		        "chromatic_aberration": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Lens-like chromatic aberration (spatial misregistration) applied to chroma latent channels (1..3) after all other adjustments. Very subtle effects are usually best (0.02–0.10)."
		        }),
		        "protect_lows": ("FLOAT", {
		            "default": 0.9,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Prevents HF detail from over-applying in low-frequency regions (reduces harshness)."
		        }),

		        # --- Soft-clip
		        "soft_clip_detail": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "Soft-limits HF detail to reduce halos/zipper edges."
		        }),
		        "soft_clip_detail_k": ("FLOAT", {
		            "default": 2.2,
		            "min": 0.5,
		            "max": 8.0,
		            "step": 0.05,
		            "tooltip": "Detail soft-clip amount. Higher = weaker limiting."
		        }),
		        "soft_clip_cfg": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "Soft-limits HF CFG injection to reduce harsh edges and background speckle."
		        }),
		        "soft_clip_cfg_k": ("FLOAT", {
		            "default": 2.0,
		            "min": 0.5,
		            "max": 8.0,
		            "step": 0.05,
		            "tooltip": "CFG soft-clip amount. Higher = weaker limiting."
		        }),

		        # --- Grain
		        "noise_scale": ("FLOAT", {
		            "default": 0.1,
		            "min": 0.0,
		            "max": 0.5,
		            "step": 0.01,
		            "tooltip": "Micrograin intensity in latent space."
		        }),
		        "grain_luma": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Luma-dependent grain: increases grain in shadows and reduces it in highlights. This primarily modulates the grain injected by noise_scale (and shaped by noise_radius)."
		        }),
		        "noise_tail": ("FLOAT", {
		            "default": -1.0,
		            "min": -1.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Tail-heaviness of the grain noise distribution. 0 = Gaussian. + = heavier tails (more occasional strong specks / grit). - = lighter tails (smoother). RMS-normalized so noise_scale stays comparable."
		        }),
		        "noise_radius": ("INT", {
		            "default": 1,
		            "min": 0,
		            "max": 16,
		            "step": 1,
		            "tooltip": "Grain correlation radius. 0=white, 1..3 often looks most photographic."
		        }),
		        "noise_flat_suppress": ("FLOAT", {
		            "default": 1.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Suppresses grain in flat regions (local smoothstep mode, always-on)."
		        }),

		        # --- Hires (bottom)
		        "hires_scale": ("FLOAT", {
		            "default": 1.0,
		            "min": 1.0,
		            "max": 4.0,
		            "step": 0.1,
		            "tooltip": "If >1, runs ONE UNet pass at upscaled latent resolution, then downsamples denoised estimates back to 1x before post-processing."
		        }),
		        "hires_strength": ("FLOAT", {
		            "default": 0.75,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "How much of the hires-derived correction to apply. 0=off, 1=full."
		        }),
		        "hires_use_importance_mask": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "If ON, applies hires correction mostly where small/foreground-ish features are detected (cheap heuristic)."
		        }),
		        "hires_mask_strength": ("FLOAT", {
		            "default": 1.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "0 = uniform hires blend, 1 = fully importance-masked blend."
		        }),

		        # --- Bottom toggles
		        "ignore_cond_timestep_range": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "If ON, strips timestep limits from conditioning ranges (more consistent behavior)."
		        }),
		        "debug": ("BOOLEAN", {
		            "default": False,
		            "tooltip": "Prints diagnostics to console."
		        }),
		    }
		}

	RETURN_TYPES = ("LATENT", )
	FUNCTION = "apply"
	CATEGORY = "latent/postprocess"

	def _convert_cached(self, cond_obj, ignore_range: bool) -> list[dict]:
		key = (id(cond_obj), bool(ignore_range))
		if key in self._conv_cache:
			return self._conv_cache[key]
		c = _maybe_convert_conditioning(cond_obj)
		if ignore_range:
			c = _strip_timestep_limits(c)
		self._conv_cache[key] = c
		if len(self._conv_cache) > 6:
			self._conv_cache.pop(next(iter(self._conv_cache)))
		return c

	def _encode_cached(self, base_model, conds: list[dict], x_in: torch.Tensor, prompt_type: str) -> list[dict]:
		key = (id(conds), str(x_in.device), str(x_in.dtype), tuple(x_in.shape), prompt_type)
		if key in self._enc_cache:
			return self._enc_cache[key]
		enc = _encode_model_conds_if_possible(base_model, conds, x_in, prompt_type)
		self._enc_cache[key] = enc
		if len(self._enc_cache) > 12:
			self._enc_cache.pop(next(iter(self._enc_cache)))
		return enc

	def _prepare_cond_batch(self, model_patcher, x_in, positive, negative, ignore_range: bool):
		if comfy_samplers is None or not hasattr(comfy_samplers, "calc_cond_batch"):
			raise RuntimeError("comfy.samplers.calc_cond_batch is unavailable. Update ComfyUI.")

		base_model = _get_base_model(model_patcher)
		model_options = _get_model_options(model_patcher, base_model)

		pos = self._convert_cached(positive, ignore_range)
		neg = self._convert_cached(negative, ignore_range)

		pos = self._encode_cached(base_model, pos, x_in, "positive")
		neg = self._encode_cached(base_model, neg, x_in, "negative")

		pos = _move_tensors(pos, x_in.device, x_in.dtype)
		neg = _move_tensors(neg, x_in.device, x_in.dtype)
		return base_model, model_options, pos, neg

	def _cond_uncond_outs_prepared(self, prepared, x_in, sigma_value: float):
		base_model, model_options, pos, neg = prepared

		sig = x_in.new_full((x_in.shape[0], ), float(sigma_value))
		outs = comfy_samplers.calc_cond_batch(base_model, [pos, neg], x_in, sig, model_options)
		if len(outs) < 2:
			raise RuntimeError("calc_cond_batch did not return [cond, uncond].")
		return base_model, outs[0], outs[1]

	def _base_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		key = (int(h), int(w), str(device), str(dtype))
		g = self._grid_cache.get(key, None)
		if g is not None:
			# Refresh LRU position.
			self._grid_cache.pop(key, None)
			self._grid_cache[key] = g
			return g
		y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
		x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
		gy, gx = torch.meshgrid(y, x, indexing="ij")
		grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
		self._grid_cache[key] = grid
		if len(self._grid_cache) > int(_GRID_CACHE_MAX):
			self._grid_cache.pop(next(iter(self._grid_cache)))
		return grid

	def _apply_chromatic_aberration(self, x: torch.Tensor, amount01: float) -> torch.Tensor:
		"""Apply a subtle, lens-like chromatic aberration to latent channels 1..3."""
		if x.shape[1] < 4:
			return x
		a = float(max(0.0, min(1.0, amount01)))
		if a <= 0.0:
			return x
		b, c, h, w = x.shape
		shift_px = a * 1.0
		dx = float(2.0 * shift_px / max(1.0, (w - 1)))
		dy = float(2.0 * shift_px / max(1.0, (h - 1)))
		grid0 = self._base_grid(h, w, x.device, x.dtype)

		g1 = grid0.clone()
		g1[..., 0].add_(dx)
		g2 = grid0.clone()
		g2[..., 0].sub_(dx)
		g3 = grid0.clone()
		g3[..., 1].add_(dy)

		c1 = F.grid_sample(x[:, 1:2], g1, mode="bilinear", padding_mode="border", align_corners=True)
		c2 = F.grid_sample(x[:, 2:3], g2, mode="bilinear", padding_mode="border", align_corners=True)
		c3 = F.grid_sample(x[:, 3:4], g3, mode="bilinear", padding_mode="border", align_corners=True)

		out = x.clone()
		out[:, 1:2] = c1
		out[:, 2:3] = c2
		out[:, 3:4] = c3
		return out

	@torch.no_grad()
	def apply(
	    self,
	    seed: int,
	    sigmas: str,
	    model,
	    latent,
	    positive,
	    negative,
	    luma_clarity: float,
	    boost_confidence: float,
	    color_drift: float,
	    color_drift_radius: int,
	    bloom_strength: float,
	    bloom_threshold: float,
	    bloom_radius: int,
	    cfg: float,
	    cfg_hf_boost: float,
	    cfg_lf_boost: float,
	    cfg_radius: int,
	    cfg_radius_flat: int,
	    cfg_radius_adaptive: bool,
	    cfg_adapt_feather: int,
	    cfg_adapt_gamma: float,
	    detail_strength: float,
	    hf_radius: int,
	    mid_strength: float,
	    detail_chroma: float,
	    chromatic_aberration: float,
	    protect_lows: float,
	    soft_clip_detail: bool,
	    soft_clip_detail_k: float,
	    soft_clip_cfg: bool,
	    soft_clip_cfg_k: float,
	    noise_scale: float,
	    noise_radius: int,
	    noise_tail: float,
	    noise_flat_suppress: float,
	    grain_luma: float,
	    hires_scale: float,
	    hires_strength: float,
	    hires_use_importance_mask: bool,
	    hires_mask_strength: float,
	    ignore_cond_timestep_range: bool,
	    debug: bool,
	    **kwargs,
	):
		_ensure_model_loaded(model)

		x_orig = latent["samples"]
		if not torch.is_tensor(x_orig):
			raise RuntimeError("LATENT['samples'] was not a tensor.")

		orig_dev = x_orig.device
		orig_dtype = x_orig.dtype

		model_dev, model_dtype = _get_model_device_dtype(model, orig_dev, orig_dtype)

		x_base = x_orig.to(device=model_dev)
		if torch.is_floating_point(x_base) and x_base.dtype != model_dtype:
			x_base = x_base.to(dtype=model_dtype)

		sigmas_list = _parse_sigmas_string(sigmas)
		sigma_values = None  # filled after base_model is available
		used_seed = _resolve_seed(int(seed))

		# Backward-compatible: old 'debug_print' input
		if bool(kwargs.get("debug_print", False)):
			debug = True

		# Base resolution latent for post-processing
		x_in = x_base

		# Cache for expensive detail masks keyed by radius (valid only for this apply() call).
		detail_mask_cache: dict[int, torch.Tensor] = {}

		# Optional hires pass: UNet at upscaled, downsample denoised back to base
		scl = float(max(1.0, min(4.0, float(hires_scale))))
		hs = float(max(0.0, min(1.0, hires_strength)))
		hm = float(max(0.0, min(1.0, hires_mask_strength)))

		# Map normalized sigma positions to actual sigma values from the model schedule.
		# Run all UNet evaluations under a single pre_run/cleanup to avoid per-step overhead.
		with _patcher_ctx(model):
			_base_for_sig = _get_base_model(model)
			sigma_values = [float(_sigma_from_ratio(_base_for_sig, r)) for r in sigmas_list]

			# Debug-only tensors (avoid extra refs when not debugging)
			out_pos = None
			out_neg = None

			use_hires = (scl > 1.0 and hs > 0.0)

			# Prepare hires tensors once
			if use_hires:
				if debug:
					print(f"[LatentDetailer] applying hires fix... scale={scl:.2f} strength={hs:.2f} sigmas={sigmas_list} mask={bool(hires_use_importance_mask)} mask_strength={hm:.2f}")

				x_hi = _upsample_latent(x_base, scl)
				hw = (x_base.shape[-2], x_base.shape[-1])

				w = hs
				if bool(hires_use_importance_mask) and hm > 0.0:
					m_imp = _hires_importance_mask(x_base)  # (B,1,H,W)
					w = hs * ((1.0 - hm) + hm * m_imp)
					w = w.clamp(0.0, 1.0)

			prepared_base = None
			prepared_hi = None
			if use_hires:
				prepared_hi = self._prepare_cond_batch(model, x_hi, positive, negative, bool(ignore_cond_timestep_range))
			else:
				prepared_base = self._prepare_cond_batch(model, x_in, positive, negative, bool(ignore_cond_timestep_range))

			den_pos_acc = None
			den_neg_acc = None

			for i, sig in enumerate(sigma_values):
				sig = float(max(1e-6, sig))

				if use_hires:
					base_model, out_pos_hi, out_neg_hi = self._cond_uncond_outs_prepared(prepared_hi, x_hi, sig)
					den_pos_hi = _calculate_denoised(base_model, x_hi, sig, out_pos_hi)
					den_neg_hi = _calculate_denoised(base_model, x_hi, sig, out_neg_hi)

					den_pos_ds = _downsample_latent_area(den_pos_hi, hw)
					den_neg_ds = _downsample_latent_area(den_neg_hi, hw)

					res_pos = den_pos_ds - x_base
					res_neg = den_neg_ds - x_base

					if torch.is_tensor(w):
						den_pos_i = x_base + res_pos * w
						den_neg_i = x_base + res_neg * w
					else:
						den_pos_i = x_base + res_pos * float(w)
						den_neg_i = x_base + res_neg * float(w)

					if debug and i == 0:
						out_pos, out_neg = out_pos_hi, out_neg_hi

				else:
					base_model, out_pos_i, out_neg_i = self._cond_uncond_outs_prepared(prepared_base, x_in, sig)
					den_pos_i = _calculate_denoised(base_model, x_in, sig, out_pos_i)
					den_neg_i = _calculate_denoised(base_model, x_in, sig, out_neg_i)

					if debug and i == 0:
						out_pos, out_neg = out_pos_i, out_neg_i

				if den_pos_acc is None:
					den_pos_acc = den_pos_i
					den_neg_acc = den_neg_i
				else:
					den_pos_acc = den_pos_acc + den_pos_i
					den_neg_acc = den_neg_acc + den_neg_i

			inv_n = 1.0 / float(max(1, len(sigma_values)))
			den_pos = den_pos_acc * inv_n
			den_neg = den_neg_acc * inv_n

		if use_hires and debug:
			print("[LatentDetailer] hires fix done.")

		# Base detail projection
		base_delta = den_pos - x_in
		base_low = _lowpass_avgpool(base_delta, int(hf_radius))
		base_hp = base_delta - base_low

		# Protect lows
		pl = float(max(0.0, min(1.0, protect_lows)))
		if pl > 0.0:
			hp_e = base_hp.abs().mean(dim=1, keepdim=True).add_(1e-6)
			d_e = base_delta.abs().mean(dim=1, keepdim=True).add_(1e-6)
			gate = hp_e / (hp_e + d_e)
			factor = gate.mul(pl).add_(1.0 - pl)
			base_hp.mul_(factor)

		# Soft clip detail
		if bool(soft_clip_detail):
			base_hp = _soft_clip_tanh(base_hp, float(soft_clip_detail_k))

		# Chroma scaling for detail/CFG injections
		cs = float(detail_chroma)
		if base_hp.shape[1] >= 4 and cs != 1.0:
			base_hp[:, 1:4].mul_(cs)

		out = x_in.clone()

		# Luma clarity
		lc = float(max(0.0, min(1.0, luma_clarity)))
		ld = None
		if lc > 0.0:
			ld = _luma_clarity_delta(den_pos, x_in, int(hf_radius), lc, mask_cache=detail_mask_cache)
			out[:, :1].add_(ld)

		# Boost confidence (channel 0 only)
		bc = float(max(0.0, min(1.0, boost_confidence)))
		bd = None
		if bc > 0.0:
			bd = _boost_confidence_delta(den_pos, x_in, int(hf_radius), bc)
			out[:, :1].add_(bd)

		# Base detail injection
		ds = float(detail_strength)
		if ds != 0.0:
			out.add_(base_hp, alpha=ds)

		# Mid/low injection
		ms = float(max(0.0, mid_strength))
		if ms != 0.0:
			out.add_(base_low, alpha=ms)

		# CFG injection
		c = float(cfg)
		cfg_scale = max(0.0, c - 1.0)
		if cfg_scale > 0.0 and (cfg_hf_boost > 0.0 or cfg_lf_boost > 0.0):
			cfg_delta = den_pos - den_neg

			if bool(cfg_radius_adaptive):
				r_flat = int(max(0, cfg_radius_flat))
				r_det = int(max(0, cfg_radius))

				mask_r = max(1, int(hf_radius))
				if mask_r in detail_mask_cache:
					m = detail_mask_cache[mask_r]
				else:
					m = _content_detail_mask_from_latent(x_in, mask_r)
					detail_mask_cache[mask_r] = m

				fr = int(max(0, cfg_adapt_feather))
				if fr > 0:
					m = _lowpass_avgpool_reflect(m, fr)

				gam = float(max(1e-3, cfg_adapt_gamma))
				if abs(gam - 1.0) > 1e-6:
					m = m.clamp(0.0, 1.0).pow(gam)
				m = m.clamp(0.0, 1.0)

				low_flat = _lowpass_avgpool(cfg_delta, r_flat)
				low_det = _lowpass_avgpool(cfg_delta, r_det)
				cfg_low = low_flat.mul(1.0 - m).add_(low_det.mul(m))
				cfg_hp = cfg_delta - cfg_low
			else:
				cfg_low = _lowpass_avgpool(cfg_delta, int(cfg_radius))
				cfg_hp = cfg_delta - cfg_low

			if bool(soft_clip_cfg):
				cfg_hp = _soft_clip_tanh(cfg_hp, float(soft_clip_cfg_k))

			if cfg_hp.shape[1] >= 4 and cs != 1.0:
				cfg_hp[:, 1:4].mul_(cs)

			hf_a = float(cfg_hf_boost) * cfg_scale
			lf_a = float(cfg_lf_boost) * cfg_scale
			if hf_a != 0.0:
				out.add_(cfg_hp, alpha=hf_a)
			if lf_a != 0.0:
				out.add_(cfg_low, alpha=lf_a)

		# Color drift
		cds = float(max(0.0, min(1.0, color_drift)))
		cd = None
		if cds > 0.0:
			cd = _color_drift_delta(
			    ref=out,
			    x_in=x_in,
			    seed=used_seed,
			    radius=int(color_drift_radius),
			    hf_radius=int(hf_radius),
			    strength=cds,
			    mask_cache=detail_mask_cache,
			)
			out.add_(cd)

		# Bloom / halation approximation (luma only)
		bs = float(max(0.0, min(1.0, bloom_strength)))
		if bs > 0.0:
			_apply_bloom_luma(out, bs, int(bloom_radius), float(bloom_threshold))

		# Grain (always local_smoothstep; noise_suppress_mode retired)
		ns = float(max(0.0, noise_scale))
		if ns > 0.0:
			n = _randn_like(out, used_seed)
			g = _bandpass_grain(n, int(noise_radius))
			nt = float(max(-1.0, min(1.0, noise_tail)))
			if nt != 0.0:
				g = _shape_noise_tail(g, nt)

			fs = float(max(0.0, min(1.0, noise_flat_suppress)))
			if fs > 0.0:
				er = max(1, int(noise_radius) * _NOISE_SUPPRESS_ENERGY_RADIUS_MULT)
				e = _local_energy_map(base_hp, er)
				t = torch.clamp((e - _NOISE_SUPPRESS_LO) / (_NOISE_SUPPRESS_HI - _NOISE_SUPPRESS_LO), 0.0, 1.0)
				allow = _smoothstep01(t)
				allow = (1.0 - fs) + fs * allow
				g = g * allow

			if _NOISE_KILL_LOWFREQ and int(noise_radius) > 0:
				rr = int(noise_radius) * _NOISE_KILL_LOWFREQ_MULT
				g = g - _lowpass_avgpool(g, rr)

			if _GRAIN_EXPOSURE_MAP:
				r = int(max(0, _GRAIN_EXPOSURE_RADIUS))
				lum = den_pos[:, :1]
				if r > 0:
					lum = _lowpass_avgpool(lum, r)
				lum = (lum - lum.mean(dim=(2, 3), keepdim=True)) / (lum.std(dim=(2, 3), keepdim=True) + 1e-6)
				grain_map = torch.sigmoid(-float(_GRAIN_EXPOSURE_STRENGTH) * lum)
				g = g * grain_map

			# Luma-dependent grain (shadows > highlights)
			gl = float(max(0.0, min(1.0, grain_luma)))
			if gl > 0.0:
				w_luma = _grain_luma_weight(den_pos, gl, int(noise_radius))
				g = g * w_luma

			if g.shape[1] >= 4 and _GRAIN_CHROMA_MODE_SEPARATE:
				g[:, 1:4].mul_(float(_GRAIN_CHROMA_STRENGTH))

			out.add_(g, alpha=ns)

		if debug:
			# out_pos/out_neg are only meaningful for debug stats; if hires was used and debug_print False,
			# they were intentionally not retained.
			d_unet = float((out_pos - out_neg).abs().mean().item()) if (out_pos is not None and out_neg is not None) else 0.0
			d_den = (den_pos - den_neg).abs().mean().item()
			d_lc = float(ld.abs().mean().item()) if ld is not None else 0.0
			d_bc = float(bd.abs().mean().item()) if bd is not None else 0.0
			d_cd = float(cd.abs().mean().item()) if cd is not None else 0.0
			print(f"[LatentDetailer] hires_scale={scl} hires_strength={hs:.2f} mask={bool(hires_use_importance_mask)} mask_strength={hm:.2f} "
			      f"| cfg={c:.3f} sigmas={','.join(f'{v:.4f}' for v in sigma_values)} lc={lc:.3f} bc={bc:.3f} drift={cds:.3f} noise={ns:.3f} "
			      f"| mean|pos-neg|={d_unet:.6g} mean|den_pos-den_neg|={d_den:.6g} "
			      f"mean|lc_delta|={d_lc:.6g} mean|bc_delta|={d_bc:.6g} mean|drift_delta|={d_cd:.6g}")

		# Chromatic aberration (spatial misregistration of chroma channels 1..3)
		ca = float(max(0.0, min(1.0, chromatic_aberration)))
		if ca > 0.0 and out.shape[1] >= 4:
			out = self._apply_chromatic_aberration(out, ca)

		# Back to original device/dtype
		out = out.to(device=orig_dev)
		if torch.is_floating_point(out) and out.dtype != orig_dtype:
			out = out.to(dtype=orig_dtype)

		out_latent = dict(latent)
		out_latent["samples"] = out
		if "noise_mask" in out_latent:
			out_latent["noise_mask"] = _normalize_noise_mask_for_samples(out_latent["noise_mask"], out)
		return (out_latent, )


class LatentDetailer(SpectralVAEDetailer):
	pass


NODE_CLASS_MAPPINGS = {
    "LatentDetailer": LatentDetailer,
    "SpectralVAEDetailer": LatentDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentDetailer": "LatentDetailer",
    "SpectralVAEDetailer": "SpectralVAEDetailer (alias)",
}
