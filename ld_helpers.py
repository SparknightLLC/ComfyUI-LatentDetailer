# custom_nodes/ComfyUI-SpectralVAE/__init__.py

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import math

import torch
import torch.nn.functional as F

try:
	import comfy.samplers as comfy_samplers
	import comfy.model_management as model_management
	import comfy.sampler_helpers as sampler_helpers
except Exception:
	comfy_samplers = None
	model_management = None
	sampler_helpers = None

# -----------------------------
# Tuned constants (hardcoded)
# -----------------------------

# Noise flat suppression (local smoothstep mode)
_NOISE_SUPPRESS_ENERGY_RADIUS_MULT = 6
_NOISE_SUPPRESS_LO = 0.20
_NOISE_SUPPRESS_HI = 0.80

# Anti-splotch: remove residual LF drift from grain
_NOISE_KILL_LOWFREQ = True
_NOISE_KILL_LOWFREQ_MULT = 4

# Grain chroma handling (reduce chroma blotches)
_GRAIN_CHROMA_MODE_SEPARATE = True
_GRAIN_CHROMA_STRENGTH = 0.35

# Exposure-dependent grain (disabled via constant)
_GRAIN_EXPOSURE_MAP = False
_GRAIN_EXPOSURE_RADIUS = 16
_GRAIN_EXPOSURE_STRENGTH = 0.8

# Adaptive CFG mask thresholds (mask energy per-image normalized => mean ~= 1.0)
_CFG_ADAPT_LO = 0.75
_CFG_ADAPT_HI = 1.35

# -----------------------------
# Color drift (noise-driven granular color drift)
# -----------------------------
_COLOR_DRIFT_MAX = 2.25
_COLOR_DRIFT_MASK_GAMMA = 0.8
_COLOR_DRIFT_BASE_ALLOW = 0.40
_COLOR_DRIFT_SOFTCLIP_K = 2.0
_COLOR_DRIFT_SEED_OFFSET = 1337

# -----------------------------
# Luma clarity (structure mid-band local contrast)
# -----------------------------
_LUMA_CLARITY_R1_MULT = 1.0
_LUMA_CLARITY_R2_MULT = 3.0
_LUMA_CLARITY_FEATHER = 2
_LUMA_CLARITY_MASK_GAMMA = 1.1
_LUMA_CLARITY_MAX = 3.5
_LUMA_CLARITY_SOFTCLIP_K = 2.2

# -----------------------------
# Boost confidence (UNet-proposed micro detail only where "confident")
# -----------------------------
_BC_CONF_LO = 0.55
_BC_CONF_HI = 1.35
_BC_EDGE_LO = 1.10
_BC_EDGE_HI = 2.20
_BC_FEATHER = 2
_BC_MAX = 1.35
_BC_SOFTCLIP_K = 2.0
_BC_BASE_ALLOW = 0.05

# -----------------------------
# Hires importance mask (cheap)
# -----------------------------
# Small-feature energy thresholds (per-image normalized => mean ~= 1)
_HIRES_IMP_LO = 0.70
_HIRES_IMP_HI = 1.90
_HIRES_IMP_FEATHER = 3

# Small feature band radii (reflect avgpool on channel 0)
_HIRES_IMP_R1 = 1
_HIRES_IMP_R2 = 4
_HIRES_IMP_ENERGY_BLUR = 2

# Center prior for "foreground-ish"
_HIRES_CENTER_SIGMA = 0.55  # smaller => more center-biased
_GRID_CACHE_MAX = 32  # max unique (H,W,device,dtype) grids cached

# Cached center prior maps
_CENTER_PRIOR_CACHE: dict[tuple[int, int, str, str], torch.Tensor] = {}

# -----------------------------
# Helpers / plumbing
# -----------------------------


def _get_base_model(model_patcher):
	m = getattr(model_patcher, "model", None)
	return m if m is not None else model_patcher


def _get_model_options(model_patcher, base_model):
	mo = getattr(model_patcher, "model_options", None)
	if isinstance(mo, dict):
		return mo
	mo = getattr(base_model, "model_options", None)
	if isinstance(mo, dict):
		return mo
	return {}


def _get_model_device_dtype(model_patcher, fallback_device, fallback_dtype):
	base_model = _get_base_model(model_patcher)
	dm = getattr(base_model, "diffusion_model", None)
	if dm is not None:
		try:
			p = next(dm.parameters())
			return p.device, p.dtype
		except Exception:
			pass
	if model_management is not None:
		try:
			return model_management.get_torch_device(), fallback_dtype
		except Exception:
			pass
	return fallback_device, fallback_dtype


def _ensure_model_loaded(model_patcher):
	if model_management is None:
		return
	try:
		model_management.load_model_gpu(model_patcher)
	except Exception:
		pass


@contextmanager
def _patcher_ctx(model_patcher):
	try:
		if hasattr(model_patcher, "pre_run"):
			model_patcher.pre_run()
		yield
	finally:
		if hasattr(model_patcher, "cleanup"):
			model_patcher.cleanup()


def _move_tensors(obj: Any, device: torch.device, dtype: torch.dtype | None):
	if torch.is_tensor(obj):
		# Prefer a single .to(device,dtype) dispatch when possible to avoid extra allocations.
		dt = dtype if (dtype is not None and torch.is_floating_point(obj)) else None
		if obj.device == device and (dt is None or obj.dtype == dt):
			return obj
		if dt is None:
			return obj.to(device=device)
		return obj.to(device=device, dtype=dt)
	if isinstance(obj, dict):
		return {k: _move_tensors(v, device, dtype) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_move_tensors(v, device, dtype) for v in obj]
	if isinstance(obj, tuple):
		return tuple(_move_tensors(v, device, dtype) for v in obj)
	return obj


def _strip_timestep_limits(cond_list: list[dict]) -> list[dict]:
	out = []
	for c in cond_list:
		if not isinstance(c, dict):
			out.append(c)
			continue
		c2 = dict(c)
		for k in ("timestep_start", "timestep_end", "start_timestep", "end_timestep"):
			c2.pop(k, None)
		out.append(c2)
	return out


def _maybe_convert_conditioning(cond):
	if cond is None:
		return []
	if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
		return cond
	if sampler_helpers is not None and hasattr(sampler_helpers, "convert_cond"):
		return sampler_helpers.convert_cond(cond)
	if comfy_samplers is not None and hasattr(comfy_samplers, "convert_cond"):
		return comfy_samplers.convert_cond(cond)
	raise RuntimeError("convert_cond not found (expected comfy.sampler_helpers.convert_cond).")


def _encode_model_conds_if_possible(base_model, conds, x_in, prompt_type: str):
	extra_conds = getattr(base_model, "extra_conds", None)
	if extra_conds is None or comfy_samplers is None or not hasattr(comfy_samplers, "encode_model_conds"):
		return conds
	try:
		return comfy_samplers.encode_model_conds(extra_conds, conds, x_in, x_in.device, prompt_type)
	except Exception:
		return conds


def _lowpass_avgpool(x: torch.Tensor, radius: int) -> torch.Tensor:
	# NOTE: original behavior intentionally preserved (avg_pool2d with zero padding).
	r = int(max(0, radius))
	if r <= 0:
		return x
	k = 2 * r + 1
	return F.avg_pool2d(x, kernel_size=k, stride=1, padding=r)


def _lowpass_avgpool_reflect(x: torch.Tensor, radius: int) -> torch.Tensor:
	# Used ONLY for masks/energy maps to avoid border artifacts.
	r = int(max(0, radius))
	if r <= 0:
		return x
	# Reflect padding requires pad < spatial size.
	max_r = max(0, min(int(x.shape[-2]), int(x.shape[-1])) - 1)
	r = min(r, max_r)
	if r <= 0:
		return x
	k = 2 * r + 1
	xp = F.pad(x, (r, r, r, r), mode="reflect")
	return F.avg_pool2d(xp, kernel_size=k, stride=1, padding=0)


def _calculate_denoised(base_model, x_in: torch.Tensor, sigma: float, model_out: torch.Tensor) -> torch.Tensor:
	ms = getattr(base_model, "model_sampling", None)
	if ms is not None and hasattr(ms, "calculate_denoised"):
		try:
			return ms.calculate_denoised(float(sigma), x_in, model_out)
		except Exception:
			sig = x_in.new_full((x_in.shape[0], ), float(sigma))
			return ms.calculate_denoised(sig, x_in, model_out)
	return x_in - model_out * float(sigma)


def _randn_like(x: torch.Tensor, seed: int) -> torch.Tensor:
	if seed < 0:
		return torch.randn_like(x)
	try:
		g = torch.Generator(device=x.device)
		g.manual_seed(int(seed))
		return torch.randn_like(x, generator=g)
	except Exception:
		torch.manual_seed(int(seed))
		return torch.randn_like(x)


def _shape_noise_tail(n: torch.Tensor, noise_tail: float) -> torch.Tensor:
	"""Reshape noise distribution tails while keeping RMS comparable.

	noise_tail in [-1..1]:
	- 0: Gaussian
	- +: heavier tails (more occasional extremes)
	- -: lighter tails (smoother, fewer extremes)
	
	Implementation notes:
	- We apply a signed power transform to adjust tail-heaviness.
	- We then renormalize per-channel RMS so `noise_scale` remains meaningful.
	"""
	t = float(max(-1.0, min(1.0, noise_tail)))
	if abs(t) < 1e-6:
		return n

	# Map tail -> exponent. +1 => p=2.0 (heavier tails), -1 => p=0.5 (lighter tails)
	p = 2.0**(t)

	x = torch.sign(n) * torch.pow(torch.abs(n) + 1e-12, p)
	return _rms_norm_(x)


def _parse_sigmas_string(sigmas_str: str) -> list[float]:
	"""Parse a comma-delimited string of normalized sigma positions in [0..1]."""
	if sigmas_str is None:
		return [0.40]
	s = str(sigmas_str).strip()
	if not s:
		return [0.40]
	parts = [p.strip() for p in s.split(",")]
	vals: list[float] = []
	for p in parts:
		if not p:
			continue
		try:
			v = float(p)
		except Exception:
			continue
		if math.isnan(v) or math.isinf(v):
			continue
		vals.append(float(max(0.0, min(1.0, v))))
	if not vals:
		return [0.40]
	return vals


def _sigma_from_ratio(base_model, r01: float) -> float:
	"""Map normalized [0..1] -> actual sigma value using the model's schedule when possible."""
	r = float(max(0.0, min(1.0, r01)))
	ms = getattr(base_model, "model_sampling", None)
	sigmas = getattr(ms, "sigmas", None) if ms is not None else None
	try:
		if sigmas is not None:
			n = int(sigmas.numel())
			if n > 0:
				idx = int(round(r * (n - 1)))
				idx = max(0, min(n - 1, idx))
				v = float(sigmas[idx].item())
				return float(max(1e-6, v))
	except Exception:
		pass

	sigma_max = None
	sigma_min = None
	if ms is not None:
		sigma_max = getattr(ms, "sigma_max", None)
		sigma_min = getattr(ms, "sigma_min", None)
	try:
		if sigma_max is not None:
			smax = float(sigma_max)
			smin = float(sigma_min) if sigma_min is not None else 1e-3
			smin = float(max(1e-6, smin))
			smax = float(max(smin, smax))
			ls = math.log(smax)
			le = math.log(smin)
			return float(math.exp(ls + (le - ls) * r))
	except Exception:
		pass

	smax = 1.0
	smin = 1e-3
	ls = math.log(smax)
	le = math.log(smin)
	return float(math.exp(ls + (le - ls) * r))


def _apply_bloom_luma(out: torch.Tensor, strength: float, radius: int, threshold01: float) -> None:
	"""Approximate bloom/halation by spreading bright highlights in luma (channel 0 only). In-place."""
	s = float(max(0.0, min(1.0, strength)))
	r = int(max(0, radius))
	t = float(max(0.0, min(1.0, threshold01)))
	if s <= 0.0 or r <= 0 or out.shape[1] < 1:
		return

	lum = out[:, :1]
	mean = lum.mean(dim=(2, 3), keepdim=True)
	std = lum.std(dim=(2, 3), keepdim=True) + 1e-6
	z = (lum - mean) / std

	ln = torch.sigmoid(z)  # 0..1
	hi = torch.relu(ln - t)

	hi_blur = _lowpass_avgpool(hi, r)
	hi_blur = (hi_blur - hi_blur.mean(dim=(2, 3), keepdim=True)) * std

	out[:, :1].add_(hi_blur, alpha=s * 0.35)


def _grain_luma_weight(den_pos: torch.Tensor, grain_luma: float, noise_radius: int) -> torch.Tensor:
	"""Return a per-pixel multiplicative weight for grain (B,1,H,W).

	This biases grain toward darker regions and reduces it in highlights.
	It primarily affects the grain injected by `noise_scale`.
	"""
	gl = float(max(0.0, min(1.0, grain_luma)))
	if gl <= 0.0 or den_pos.ndim != 4:
		# Safe fallback
		b = int(den_pos.shape[0]) if hasattr(den_pos, "shape") else 1
		h = int(den_pos.shape[-2]) if hasattr(den_pos, "shape") else 64
		w = int(den_pos.shape[-1]) if hasattr(den_pos, "shape") else 64
		return den_pos.new_ones((b, 1, h, w))

	c = int(min(4, den_pos.shape[1]))
	lum = den_pos[:, :c].mean(dim=1, keepdim=True)

	# Blur to behave more like exposure-based grain rather than pixel-level chatter.
	r = int(max(0, noise_radius))
	blur_r = int(min(24, max(2, r * 4)))  # tie to grain correlation, but clamp for speed
	if blur_r > 0:
		lum = _lowpass_avgpool(lum, blur_r)

	mean = lum.mean(dim=(2, 3), keepdim=True)
	std = lum.std(dim=(2, 3), keepdim=True) + 1e-6
	z = (lum - mean) / std

	# Exponential curve: z < 0 (darker) => weight > 1 ; z > 0 => weight < 1
	w_base = torch.exp(-1.75 * z)
	w_base = torch.clamp(w_base, 0.25, 4.0)

	w = (1.0 - gl) + gl * w_base
	return w


def _resolve_seed(seed: int) -> int:
	if seed >= 0:
		return int(seed)
	return int(torch.randint(0, 2**31 - 1, (1, )).item())


def _rms_norm_(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	var = x.pow(2).mean(dim=(2, 3), keepdim=True)
	return x.mul_(torch.rsqrt(var + eps))


def _bandpass_grain(noise: torch.Tensor, r: int) -> torch.Tensor:
	r = int(max(0, r))
	if r == 0:
		return _rms_norm_(noise)
	lp1 = _lowpass_avgpool(noise, r)
	lp2 = _lowpass_avgpool(noise, r * 2)
	band = lp1 - lp2
	return _rms_norm_(band)


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
	return x * x * (3.0 - 2.0 * x)


def _local_energy_map(hp: torch.Tensor, r: int) -> torch.Tensor:
	e = hp.pow(2).mean(dim=1, keepdim=True)
	if r > 0:
		e = _lowpass_avgpool_reflect(e, r)
	e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)
	return e


def _content_detail_mask_from_latent(x: torch.Tensor, r: int) -> torch.Tensor:
	rr = int(max(0, r))
	l = x[:, :1]

	dx = l[..., 1:] - l[..., :-1]
	dy = l[..., 1:, :] - l[..., :-1, :]

	dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
	dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

	gm = dx.abs_().add_(dy.abs_())

	e = gm
	if rr > 0:
		e = _lowpass_avgpool_reflect(e, rr)
	e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)

	t = torch.clamp((e - _CFG_ADAPT_LO) / (_CFG_ADAPT_HI - _CFG_ADAPT_LO), 0.0, 1.0)
	return _smoothstep01(t)


def _soft_clip_tanh(x: torch.Tensor, k: float) -> torch.Tensor:
	kk = float(max(1e-6, k))
	return torch.tanh(x * kk) / kk


def _rms_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	return torch.sqrt(x.pow(2).mean(dim=(1, 2, 3), keepdim=True) + eps)


def _upsample_latent(x: torch.Tensor, scale: float) -> torch.Tensor:
	s = float(scale)
	if not (s > 1.0):
		return x
	h, w = int(x.shape[-2]), int(x.shape[-1])
	th = max(1, int(round(h * s)))
	tw = max(1, int(round(w * s)))
	if th == h and tw == w:
		return x
	return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)


def _downsample_latent_area(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
	h, w = int(size_hw[0]), int(size_hw[1])
	if x.shape[-2] == h and x.shape[-1] == w:
		return x
	return F.interpolate(x, size=(h, w), mode="area")


def _normalize_noise_mask_for_samples(noise_mask: Any, samples: torch.Tensor) -> Any:
	"""Normalize noise_mask shape to match latent sample spatial/batch dimensions."""
	if not torch.is_tensor(noise_mask):
		return noise_mask
	if not torch.is_tensor(samples) or samples.ndim < 4:
		return noise_mask

	b = int(samples.shape[0])
	h = int(samples.shape[-2])
	w = int(samples.shape[-1])

	m = noise_mask
	if m.ndim == 2:
		m = m.reshape((1, 1, m.shape[-2], m.shape[-1]))
	elif m.ndim == 3:
		m = m.reshape((m.shape[0], 1, m.shape[-2], m.shape[-1]))
	elif m.ndim >= 4:
		m = m.reshape((m.shape[0], m.shape[1], m.shape[-2], m.shape[-1]))
	else:
		return noise_mask

	target_device = samples.device
	if m.device != target_device:
		m = m.to(device=target_device)

	if m.shape[-2] != h or m.shape[-1] != w:
		mf = m.float() if not torch.is_floating_point(m) else m
		m = F.interpolate(mf, size=(h, w), mode="bilinear", align_corners=False)

	if m.shape[0] != b:
		if m.shape[0] == 1:
			m = m.repeat((b, ) + (1,) * (m.ndim - 1))
		else:
			m = m[:b]

	return m.contiguous()


def _center_prior(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	key = (int(h), int(w), str(device), str(dtype))
	cached = _CENTER_PRIOR_CACHE.get(key)
	if cached is not None:
		return cached

	yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1)
	xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w)
	r2 = xx * xx + yy * yy
	sig = float(max(1e-6, _HIRES_CENTER_SIGMA))
	cp = torch.exp(-r2 / (2.0 * sig * sig))  # (1,1,H,W)

	_CENTER_PRIOR_CACHE[key] = cp
	if len(_CENTER_PRIOR_CACHE) > 12:
		_CENTER_PRIOR_CACHE.pop(next(iter(_CENTER_PRIOR_CACHE)))
	return cp


def _hires_importance_mask(x_base: torch.Tensor) -> torch.Tensor:
	"""
	Cheap importance mask in [0..1]:
	- small-feature dense (band energy)
	- likely foreground-ish (center prior)
	"""
	l = x_base[:, :1]

	lp1 = _lowpass_avgpool_reflect(l, int(_HIRES_IMP_R1))
	lp2 = _lowpass_avgpool_reflect(l, int(_HIRES_IMP_R2))
	band = lp1 - lp2

	e = band.pow(2)
	e = _lowpass_avgpool_reflect(e, int(_HIRES_IMP_ENERGY_BLUR))
	e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)

	t = torch.clamp((e - _HIRES_IMP_LO) / (_HIRES_IMP_HI - _HIRES_IMP_LO), 0.0, 1.0)
	m = _smoothstep01(t)

	cp = _center_prior(m.shape[-2], m.shape[-1], device=m.device, dtype=m.dtype)
	m = m * cp

	if _HIRES_IMP_FEATHER > 0:
		m = _lowpass_avgpool_reflect(m, int(_HIRES_IMP_FEATHER))

	return m.clamp(0.0, 1.0)


def _color_drift_delta(
    ref: torch.Tensor,
    x_in: torch.Tensor,
    seed: int,
    radius: int,
    hf_radius: int,
    strength: float,
    mask_cache: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
	s = float(max(0.0, min(1.0, strength)))
	if s <= 0.0:
		return ref.mul(0.0)

	C = ref.shape[1]
	if C < 2:
		return ref.mul(0.0)

	c1 = 1
	c2 = min(4, C)
	# Only generate noise for chroma channels we actually modify.
	n = _randn_like(ref[:, c1:c2], int(seed) + int(_COLOR_DRIFT_SEED_OFFSET))
	dr = _bandpass_grain(n, int(max(0, radius)))

	dr = _soft_clip_tanh(dr, float(_COLOR_DRIFT_SOFTCLIP_K))

	dr = dr - dr.mean(dim=1, keepdim=True)  # no per-pixel tint
	dr = dr - dr.mean(dim=(2, 3), keepdim=True)  # no global cast
	dr = dr * torch.rsqrt(_rms_per_image(dr) + 1e-6)

	mr = max(1, int(hf_radius))
	if mask_cache is not None and mr in mask_cache:
		m = mask_cache[mr]
	else:
		m = _content_detail_mask_from_latent(x_in, mr)
		if mask_cache is not None:
			mask_cache[mr] = m
	m = m.clamp(0.0, 1.0).pow(float(_COLOR_DRIFT_MASK_GAMMA))
	gate = float(_COLOR_DRIFT_BASE_ALLOW) + (1.0 - float(_COLOR_DRIFT_BASE_ALLOW)) * m

	amt = float(_COLOR_DRIFT_MAX) * s

	delta = ref.mul(0.0)
	delta[:, c1:c2] = dr * gate * amt
	return delta


def _luma_clarity_delta(
    den_pos: torch.Tensor,
    x_in: torch.Tensor,
    hf_radius: int,
    strength: float,
    mask_cache: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
	s = float(max(0.0, min(1.0, strength)))
	if s <= 0.0:
		return den_pos[:, :1].mul(0.0)

	l = den_pos[:, :1]
	r1 = max(1, int(round(float(hf_radius) * _LUMA_CLARITY_R1_MULT)))
	r2 = max(r1 + 1, int(round(float(hf_radius) * _LUMA_CLARITY_R2_MULT)))

	lp1 = _lowpass_avgpool_reflect(l, r1)
	lp2 = _lowpass_avgpool_reflect(l, r2)
	band = lp1 - lp2

	band = _soft_clip_tanh(band, float(_LUMA_CLARITY_SOFTCLIP_K))
	band = band * torch.rsqrt(_rms_per_image(band) + 1e-6)

	mr = max(1, int(hf_radius))
	if mask_cache is not None and mr in mask_cache:
		m = mask_cache[mr]
	else:
		m = _content_detail_mask_from_latent(x_in, mr)
		if mask_cache is not None:
			mask_cache[mr] = m
	if _LUMA_CLARITY_FEATHER > 0:
		m = _lowpass_avgpool_reflect(m, int(_LUMA_CLARITY_FEATHER))
	m = m.clamp(0.0, 1.0).pow(float(_LUMA_CLARITY_MASK_GAMMA))

	gate = 0.15 + 0.85 * m
	amt = float(_LUMA_CLARITY_MAX) * s
	return band * gate * amt


def _boost_confidence_delta(
    den_pos: torch.Tensor,
    x_in: torch.Tensor,
    hf_radius: int,
    strength: float,
) -> torch.Tensor:
	s = float(max(0.0, min(1.0, strength)))
	if s <= 0.0:
		return den_pos[:, :1].mul(0.0)

	r = int(max(1, hf_radius))
	d0 = (den_pos[:, :1] - x_in[:, :1])

	low = _lowpass_avgpool(d0, r)
	hp = d0 - low
	hp = _soft_clip_tanh(hp, float(_BC_SOFTCLIP_K))

	conf = _local_energy_map(hp, max(1, r))
	t = torch.clamp((conf - _BC_CONF_LO) / (_BC_CONF_HI - _BC_CONF_LO), 0.0, 1.0)
	conf_w = _smoothstep01(t)
	conf_w = float(_BC_BASE_ALLOW) + (1.0 - float(_BC_BASE_ALLOW)) * conf_w

	l = x_in[:, :1]
	dx = l[..., 1:] - l[..., :-1]
	dy = l[..., 1:, :] - l[..., :-1, :]
	dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
	dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
	gm = dx.abs_().add_(dy.abs_())

	edge_e = gm.pow(2)
	edge_e = _lowpass_avgpool_reflect(edge_e, max(1, r))
	edge_e = edge_e / (edge_e.mean(dim=(2, 3), keepdim=True) + 1e-6)

	et = torch.clamp((edge_e - _BC_EDGE_LO) / (_BC_EDGE_HI - _BC_EDGE_LO), 0.0, 1.0)
	edge_w = 1.0 - _smoothstep01(et)

	w = conf_w * edge_w
	if _BC_FEATHER > 0:
		w = _lowpass_avgpool_reflect(w, int(_BC_FEATHER))

	hp = hp * torch.rsqrt(_rms_per_image(hp) + 1e-6)

	amt = float(_BC_MAX) * s
	return hp * w * amt


# -----------------------------
# Node
# -----------------------------


__all__ = [
	name for name in globals().keys()
	if (
		name.startswith("_")
		or name in ("comfy_samplers", "model_management", "sampler_helpers")
	)
	and not name.startswith("__")
]


