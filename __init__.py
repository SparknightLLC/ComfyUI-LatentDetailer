# custom_nodes/ComfyUI-SpectralVAE/__init__.py

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

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
		t = obj.to(device=device)
		if dtype is not None and torch.is_floating_point(t) and t.dtype != dtype:
			t = t.to(dtype=dtype)
		return t
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
	r = int(max(0, radius))
	if r <= 0:
		return x
	k = 2 * r + 1
	return F.avg_pool2d(x, kernel_size=k, stride=1, padding=r)


def _calculate_denoised(base_model, x_in: torch.Tensor, sigma: float, model_out: torch.Tensor) -> torch.Tensor:
	ms = getattr(base_model, "model_sampling", None)
	if ms is not None and hasattr(ms, "calculate_denoised"):
		sig = x_in.new_full((x_in.shape[0], ), float(sigma))
		try:
			return ms.calculate_denoised(float(sigma), x_in, model_out)
		except Exception:
			return ms.calculate_denoised(sig, x_in, model_out)
	# fallback (eps-pred assumption)
	return x_in - model_out * float(sigma)


def _randn_like(x: torch.Tensor, seed: int) -> torch.Tensor:
	if seed < 0:
		return torch.randn_like(x)
	try:
		g = torch.Generator(device=x.device)
		g.manual_seed(int(seed))
		return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
	except TypeError:
		torch.manual_seed(int(seed))
		return torch.randn_like(x)


def _resolve_seed(seed: int) -> int:
	if seed >= 0:
		return int(seed)
	return int(torch.randint(0, 2**31 - 1, (1, )).item())


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	var = x.pow(2).mean(dim=(2, 3), keepdim=True)
	return x * torch.rsqrt(var + eps)


def _bandpass_grain(noise: torch.Tensor, r: int) -> torch.Tensor:
	"""
    Grain size that actually changes:
      - r=0: white noise
      - r>0: band-pass correlated noise centered around scale ~r
    """
	r = int(max(0, r))
	if r == 0:
		return _rms_norm(noise)

	lp1 = _lowpass_avgpool(noise, r)
	lp2 = _lowpass_avgpool(noise, r * 2)
	band = lp1 - lp2
	return _rms_norm(band)


# -----------------------------
# Node
# -----------------------------


class SpectralVAEDetailer:
	"""
    SpectralVAEDetailer
    - 1x UNet forward at chosen sigma
    - Base detail projection from den_pos
    - Dedicated CFG injection from (den_pos - den_neg)
    - Latent micrograin injection with true radius + flat suppression
    """

	def __init__(self):
		self._conv_cache = {}
		self._enc_cache = {}

	@classmethod
	def INPUT_TYPES(cls):
		return {
		    "required": {
		        # --- Top controls
		        "seed": ("INT", {
		            "default": -1,
		            "min": -1,
		            "max": 2**31 - 1,
		            "step": 1
		        }),

		        # --- Main inputs
		        "model": ("MODEL", ),
		        "latent": ("LATENT", ),
		        "positive": ("CONDITIONING", ),
		        "negative": ("CONDITIONING", ),

		        # --- CFG group (final defaults)
		        "cfg": ("FLOAT", {
		            "default": 7.0,
		            "min": 0.0,
		            "max": 10.0,
		            "step": 0.05
		        }),
		        "cfg_hf_boost": ("FLOAT", {
		            "default": 5.0,
		            "min": 0.0,
		            "max": 5.0,
		            "step": 0.05
		        }),
		        "cfg_lf_boost": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.02
		        }),
		        "cfg_radius": ("INT", {
		            "default": 5,
		            "min": 0,
		            "max": 64,
		            "step": 1
		        }),

		        # --- Core look (final defaults)
		        "sigma": ("FLOAT", {
		            "default": 0.4,
		            "min": 0.001,
		            "max": 50.0,
		            "step": 0.001
		        }),
		        "detail_strength": ("FLOAT", {
		            "default": 0.65,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.01
		        }),
		        "hf_radius": ("INT", {
		            "default": 8,
		            "min": 0,
		            "max": 64,
		            "step": 1
		        }),
		        "mid_strength": ("FLOAT", {
		            "default": 0.05,
		            "min": 0.0,
		            "max": 0.5,
		            "step": 0.01
		        }),
		        "chroma_strength": ("FLOAT", {
		            "default": 0.1,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.01
		        }),
		        "protect_lows": ("FLOAT", {
		            "default": 0.9,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01
		        }),

		        # --- Grain (final defaults)
		        "noise_scale": ("FLOAT", {
		            "default": 0.2,
		            "min": 0.0,
		            "max": 0.5,
		            "step": 0.01
		        }),
		        "noise_radius": ("INT", {
		            "default": 1,
		            "min": 0,
		            "max": 16,
		            "step": 1
		        }),
		        "noise_flat_suppress": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01
		        }),

		        # --- Toggles at bottom
		        "ignore_cond_timestep_range": ("BOOLEAN", {
		            "default": True
		        }),
		        "debug_print": ("BOOLEAN", {
		            "default": False
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

	def _cond_uncond_outs(self, model_patcher, x_in, sigma_value: float, positive, negative, ignore_range: bool):
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

		sig = x_in.new_full((x_in.shape[0], ), float(sigma_value))
		outs = comfy_samplers.calc_cond_batch(base_model, [pos, neg], x_in, sig, model_options)
		if len(outs) < 2:
			raise RuntimeError("calc_cond_batch did not return [cond, uncond].")
		return base_model, outs[0], outs[1]

	@torch.no_grad()
	def apply(
	    self,
	    seed: int,
	    model,
	    latent,
	    positive,
	    negative,
	    cfg: float,
	    cfg_hf_boost: float,
	    cfg_lf_boost: float,
	    cfg_radius: int,
	    sigma: float,
	    detail_strength: float,
	    hf_radius: int,
	    mid_strength: float,
	    chroma_strength: float,
	    protect_lows: float,
	    noise_scale: float,
	    noise_radius: int,
	    noise_flat_suppress: float,
	    ignore_cond_timestep_range: bool,
	    debug_print: bool,
	):
		_ensure_model_loaded(model)

		x_orig = latent["samples"]
		if not torch.is_tensor(x_orig):
			raise RuntimeError("LATENT['samples'] was not a tensor.")

		orig_dev = x_orig.device
		orig_dtype = x_orig.dtype

		model_dev, model_dtype = _get_model_device_dtype(model, orig_dev, orig_dtype)

		x = x_orig.to(device=model_dev)
		if torch.is_floating_point(x) and x.dtype != model_dtype:
			x = x.to(dtype=model_dtype)

		sig = float(max(1e-6, sigma))
		used_seed = _resolve_seed(int(seed))

		x_in = x

		with _patcher_ctx(model):
			base_model, out_pos, out_neg = self._cond_uncond_outs(model, x_in, sig, positive, negative, bool(ignore_cond_timestep_range))

		den_pos = _calculate_denoised(base_model, x_in, sig, out_pos)
		den_neg = _calculate_denoised(base_model, x_in, sig, out_neg)

		# Base detail projection uses den_pos
		base_delta = den_pos - x_in

		base_low = _lowpass_avgpool(base_delta, int(hf_radius))
		base_hp = base_delta - base_low

		pl = float(max(0.0, min(1.0, protect_lows)))
		if pl > 0.0:
			hp_e = base_hp.abs().mean(dim=1, keepdim=True) + 1e-6
			d_e = base_delta.abs().mean(dim=1, keepdim=True) + 1e-6
			gate = hp_e / (hp_e + d_e)
			base_hp = base_hp * gate.lerp(torch.ones_like(gate), 1.0 - pl)

		cs = float(chroma_strength)
		if base_hp.shape[1] >= 4 and cs != 1.0:
			hp_struct = base_hp[:, :1]
			hp_other = base_hp[:, 1:4] * cs
			base_hp = torch.cat([hp_struct, hp_other], dim=1)

		out = x + base_hp * float(detail_strength) + base_low * float(max(0.0, mid_strength))

		# Dedicated CFG injection (visible)
		c = float(cfg)
		cfg_scale = max(0.0, c - 1.0)
		if cfg_scale > 0.0 and (cfg_hf_boost > 0.0 or cfg_lf_boost > 0.0):
			cfg_delta = (den_pos - den_neg)
			cfg_low = _lowpass_avgpool(cfg_delta, int(cfg_radius))
			cfg_hp = cfg_delta - cfg_low

			if cfg_hp.shape[1] >= 4 and cs != 1.0:
				hp_struct = cfg_hp[:, :1]
				hp_other = cfg_hp[:, 1:4] * cs
				cfg_hp = torch.cat([hp_struct, hp_other], dim=1)

			out = out + cfg_hp * float(cfg_hf_boost) * cfg_scale + cfg_low * float(cfg_lf_boost) * cfg_scale

		# Micrograin (visible radius + meaningful flat suppression)
		ns = float(max(0.0, noise_scale))
		if ns > 0.0:
			n = _randn_like(out, used_seed)
			g = _bandpass_grain(n, int(noise_radius))

			fs = float(max(0.0, min(1.0, noise_flat_suppress)))
			if fs > 0.0:
				e = base_hp.abs().mean(dim=1, keepdim=True)
				e = e / (e.mean() + 1e-6)
				allow = torch.clamp(e, 0.0, 1.0).pow(2.0)
				allow = (1.0 - fs) + fs * allow
				g = g * allow

			if g.shape[1] >= 4 and cs != 1.0:
				g_struct = g[:, :1]
				g_other = g[:, 1:4] * cs
				g = torch.cat([g_struct, g_other], dim=1)

			out = out + g * ns

		if debug_print:
			d_unet = (out_pos - out_neg).abs().mean().item()
			d_den = (den_pos - den_neg).abs().mean().item()
			print(f"[SpectralVAEDetailer] used_seed={used_seed} cfg={c:.3f} sigma={sig:.4f} noise_scale={ns:.3f} "
			      f"| mean|pos-neg|={d_unet:.6g} mean|den_pos-den_neg|={d_den:.6g}")

		out = out.to(device=orig_dev)
		if torch.is_floating_point(out) and out.dtype != orig_dtype:
			out = out.to(dtype=orig_dtype)

		out_latent = dict(latent)
		out_latent["samples"] = out
		return (out_latent, )


NODE_CLASS_MAPPINGS = {
    "SpectralVAEDetailer": SpectralVAEDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralVAEDetailer": "SpectralVAEDetailer",
}
