"""
Parametric D3FL* (single-file scaffold)

This file defines a **parameterized** D3FL* defense that you can plug into an
existing FL simulator/training loop.

D3FL* knobs (suggested paper notation):
  - b : quantization bits (or, equivalently, a quantization scale)
  - q : audit sketch count (integrity strength)
  - s : minimum cohort size (min #participants per round)
  - tau : audit tolerance threshold (integer) for robust verification
  - rho : audit frequency (e.g., 1.0 audit every round; 0.2 audit 1/5 rounds)

What D3FL* provides (conceptually):
  - SA (secure aggregation): server only learns aggregate sum/mean (confidentiality)
  - AUD (audit): clients can verify server did not deviate from pure aggregation (integrity)
  - Optional: rule "reject if participants < s" (prevents privacy collapse when n is too small)

This file intentionally DOES NOT implement:
  - model training / local update computation
  - network transport / relay
  - dropout-robust SecAgg protocols
  - cryptographic key exchange
  - server orchestration

Instead, it defines clear interfaces you call from your system:
  - get_round_policy(...)  -> decide whether to audit this round; enforce min cohort size
  - client_prepare_message(...) -> quantize + compute audit sketches + mask (hook points)
  - server_aggregate(...)  -> aggregate masked updates and sketches (hook points)
  - client_verify_broadcast(...) -> verify audit equations (with tolerance tau)
  - communication_accounting(...) -> compute comm bits/param/client for reporting

You can implement your own SA backend (pairwise masks, CCS'17, etc.) and pass it in.

Author: (your name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import hashlib
import random

import torch


# ---------------------------
# External interfaces (to be implemented by your codebase)
# ---------------------------

class SABackend:
    """
    External Secure Aggregation backend interface.

    You must implement these methods in your system:
      - mask_vector: mask a vector update for secure aggregation
      - mask_scalar(s): mask audit sketches / scalar metrics (optional)

    This file assumes cross-silo stable participation; for cross-device/dropout,
    use a proper SecAgg backend and adapt accordingly.
    """
    def mask_vector(self, cid: int, round_idx: int, vec_int: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mask_scalars(self, cid: int, round_idx: int, scalars: List[int], tag: str) -> List[int]:
        raise NotImplementedError

    def mask_scalar(self, cid: int, round_idx: int, value: int, tag: str) -> int:
        raise NotImplementedError


class ModelAdapter:
    """
    External model interface.
    Your code should produce per-client updates (delta) as a 1D float tensor.

    Typical usage:
      delta_float: torch.Tensor = model_adapter.compute_client_update(...)
    """
    def flatten_update(self, delta_float: torch.Tensor) -> torch.Tensor:
        """Return a 1D float tensor representing model update."""
        raise NotImplementedError


# ---------------------------
# D3FL* configuration
# ---------------------------

@dataclass
class D3FLConfig:
    # --- Main paper knobs ---
    b: int = 32                 # quantization bits (8/16/32)
    q: int = 8                  # audit sketch count
    s: int = 2                  # minimum cohort size (min participants per round)
    tau: int = 0                # audit tolerance (integer units in quantized domain)
    rho: float = 1.0            # audit frequency in (0,1]; 1.0 = every round

    # --- Quantization options ---
    # Option A: derived scale from bits via symmetric uniform quantization
    # Option B: directly set scale; if provided, overrides b-based scale.
    quant_scale: Optional[float] = None
    # Maximum absolute value expected for update elements (for b-based quant)
    clip_norm: Optional[float] = None  # if set, clip update L2 norm before quant

    # --- RNG control ---
    seed: int = 0


# ---------------------------
# Helper: Rademacher sketches for audit
# ---------------------------

def _seed_to_int64(seed_t: int, j: int) -> int:
    h = hashlib.sha256(f"{seed_t}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def rademacher_dot(seed_t: int, j: int, vec_int: torch.Tensor) -> int:
    """
    Compute dot(r, vec_int), where r in {-1,+1}^d is generated deterministically from (seed_t, j).
    vec_int must be int64.
    """
    assert vec_int.dtype == torch.int64
    d = vec_int.numel()
    g = torch.Generator(device=vec_int.device)
    g.manual_seed(_seed_to_int64(seed_t, j))
    r01 = torch.randint(0, 2, size=(d,), generator=g, device=vec_int.device, dtype=torch.int64)
    r = r01 * 2 - 1
    return int((r * vec_int).sum().item())


# ---------------------------
# Helper: quantization
# ---------------------------

def _l2_clip(delta: torch.Tensor, clip_norm: float) -> torch.Tensor:
    if clip_norm is None:
        return delta
    norm = float(torch.norm(delta).item()) + 1e-12
    if norm <= clip_norm:
        return delta
    return delta * (clip_norm / norm)


def choose_quant_scale(delta_float: torch.Tensor, b: int) -> float:
    """
    Choose a symmetric quantization scale so that int range roughly fits delta.
    For reproducibility, you may prefer a fixed scale; otherwise this is a simple heuristic.

    For b bits signed: max int is 2^(b-1)-1.
    scale = max_abs / max_int, so int = round(delta / scale).
    """
    max_int = (2 ** (b - 1)) - 1
    max_abs = float(delta_float.abs().max().item()) + 1e-12
    return max_abs / max_int


def quantize_to_int(delta_float: torch.Tensor, b: int, scale: Optional[float], clip_norm: Optional[float]) -> Tuple[torch.Tensor, float]:
    """
    Return (delta_int64, used_scale).
    delta_int = round(delta_float / scale).
    """
    delta = _l2_clip(delta_float, clip_norm)
    used_scale = scale if scale is not None else choose_quant_scale(delta, b)
    delta_int = torch.round(delta / used_scale).to(torch.int64)
    return delta_int, used_scale


def dequantize_from_int(delta_int: torch.Tensor, scale: float) -> torch.Tensor:
    return delta_int.to(torch.float32) * float(scale)


# ---------------------------
# D3FL* core
# ---------------------------

class D3FL:
    """
    Parameterized D3FL* defense core.

    Integration points (external):
      - Provide a SABackend to mask vectors/scalars for secure aggregation.
      - Your FL runner supplies:
          - per-client delta updates (float tensors)
          - participant list for the round
          - server broadcast delta_agg_int (sum) and sketch_sum (sum)
          - seed_t per round (public)
      - Optionally provide weights (e.g., sample-size weights) at aggregation time.

    Notes:
      - This class assumes a *sum* aggregation in the quantized domain.
      - If you need weighted mean, apply weights consistently (and account for them in sketches).
        For clarity, we implement unweighted mean as: delta_mean = delta_sum / m.
    """

    def __init__(self, cfg: D3FLConfig, sa_backend: SABackend):
        self.cfg = cfg
        self.sa = sa_backend
        self.rng = random.Random(cfg.seed)

    # ---------------------------
    # Round policy: audit scheduling and minimum cohort size
    # ---------------------------
    def get_round_policy(self, round_idx: int, num_participants: int) -> Dict[str, Any]:
        """
        Decide if the round is allowed and whether to audit.

        Returns:
          {
            "allow_round": bool,
            "audit_this_round": bool,
          }
        """
        allow = num_participants >= int(self.cfg.s)
        # audit with frequency rho (deterministic by seed+round for reproducibility)
        audit = False
        if allow and self.cfg.q > 0 and self.cfg.rho > 0:
            # deterministic Bernoulli
            h = hashlib.sha256(f"{self.cfg.seed}|audit|{round_idx}".encode()).digest()
            u = int.from_bytes(h[:4], "little") / 2**32
            audit = (u < float(self.cfg.rho))
        return {"allow_round": allow, "audit_this_round": audit}

    # ---------------------------
    # Client-side: prepare masked message
    # ---------------------------
    def client_prepare_message(
        self,
        *,
        cid: int,
        round_idx: int,
        seed_t: int,
        delta_float_flat: torch.Tensor,
        audit_this_round: bool,
        tag_prefix: str = "d3fl",
    ) -> Dict[str, Any]:
        """
        Client prepares a message for secure aggregation.

        Inputs:
          cid: client id
          round_idx: current FL round
          seed_t: public round seed for sketches
          delta_float_flat: 1D float tensor of model update (w_after - w_before)
          audit_this_round: whether to include audit sketches

        Output (to be sent via your transport/relay):
          {
            "masked_update": torch.int64 tensor,
            "masked_sketches": list[int] length q (or empty),
            "quant_scale": float used_scale (optional: report to server/client; in practice fix it),
          }

        IMPORTANT:
          - In real systems, all clients should use the same quant_scale (fixed) for consistency.
            For research, you can also fix cfg.quant_scale globally.
        """
        delta_int, used_scale = quantize_to_int(
            delta_float_flat,
            b=int(self.cfg.b),
            scale=self.cfg.quant_scale,
            clip_norm=self.cfg.clip_norm,
        )

        # audit sketches (unmasked)
        sketches: List[int] = []
        if audit_this_round and int(self.cfg.q) > 0:
            for j in range(int(self.cfg.q)):
                sketches.append(rademacher_dot(seed_t, j, delta_int))

        # SA masking
        masked_update = self.sa.mask_vector(cid, round_idx, delta_int)
        masked_sketches = self.sa.mask_scalars(cid, round_idx, sketches, tag=f"{tag_prefix}:sk") if sketches else []

        return {
            "masked_update": masked_update,
            "masked_sketches": masked_sketches,
            "quant_scale": float(used_scale),
        }

    # ---------------------------
    # Server-side: aggregate masked messages (sum) + prepare broadcast
    # ---------------------------
    def server_aggregate(
        self,
        *,
        round_idx: int,
        seed_t: int,
        batch: Sequence[Dict[str, Any]],
        audit_this_round: bool,
        weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate masked updates/sketches. Masks cancel in sum if SA backend is correct.

        Inputs:
          batch: list of dicts with keys "masked_update", "masked_sketches"
          weights: OPTIONAL. If you want weighted mean, you must:
            - either incorporate weights into client updates before masking
            - or scale masked_update accordingly (hard in int domain)
          For simplicity, we assume equal weights and compute mean at apply time.

        Output (broadcast to clients):
          {
            "agg_update_int": int64 tensor (sum of updates),
            "sketch_sum_int": int64 tensor length q (sum of sketches) if audited,
            "num_clients": int,
            "seed_t": int,
          }
        """
        assert len(batch) > 0, "empty batch"
        updates = [m["masked_update"].to(torch.int64) for m in batch]
        agg_update_int = torch.stack(updates, dim=0).sum(dim=0)

        out: Dict[str, Any] = {
            "agg_update_int": agg_update_int,
            "num_clients": int(len(batch)),
            "seed_t": int(seed_t),
        }

        if audit_this_round and int(self.cfg.q) > 0:
            # sum sketches (each msg has list[int] length q)
            q = int(self.cfg.q)
            sk_sum = torch.zeros(q, dtype=torch.int64)
            for m in batch:
                sk = m.get("masked_sketches", [])
                if len(sk) != q:
                    raise ValueError("missing sketches in audited round")
                sk_sum += torch.tensor(sk, dtype=torch.int64)
            out["sketch_sum_int"] = sk_sum
            out["q"] = q

        return out

    # ---------------------------
    # Client-side: verify broadcast (AUD)
    # ---------------------------
    def client_verify_broadcast(
        self,
        *,
        seed_t: int,
        agg_update_int: torch.Tensor,
        sketch_sum_int: Optional[torch.Tensor],
        audit_this_round: bool,
    ) -> bool:
        """
        Verify audit equations:
          dot(r_j, agg_update_int) == sketch_sum_int[j]  (within tolerance tau)

        If audit_this_round is False, returns True.
        """
        if not audit_this_round:
            return True
        if sketch_sum_int is None:
            return False
        q = int(self.cfg.q)
        tau = int(self.cfg.tau)

        agg = agg_update_int.to(torch.int64)
        sks = sketch_sum_int.to(torch.int64)
        if sks.numel() != q:
            return False

        for j in range(q):
            lhs = rademacher_dot(seed_t, j, agg)
            rhs = int(sks[j].item())
            if abs(lhs - rhs) > tau:
                return False
        return True

    # ---------------------------
    # Apply aggregated update to a model (external)
    # ---------------------------
    def dequantized_mean_update(
        self,
        *,
        agg_update_int: torch.Tensor,
        used_scale: float,
        num_clients: int,
    ) -> torch.Tensor:
        """
        Convert aggregated int update to float mean update:
          delta_mean = (agg_update_int * scale) / num_clients

        Your runner should then apply this delta to the global model parameters.

        NOTE: If you do weighted mean, adjust here accordingly.
        """
        delta_sum = dequantize_from_int(agg_update_int, used_scale)
        return delta_sum / max(1, int(num_clients))

    # ---------------------------
    # Communication accounting (paper-ready)
    # ---------------------------
    def communication_bits_per_param(self, *, d: int) -> float:
        """
        Compute an *approximate* communication cost in bits per parameter per client per round:
          - update: b bits per parameter
          - audit: q sketches each is 64 bits, amortized over d params if sent per round

        This is a reporting metric for your tables/plots (not a protocol guarantee).
        """
        b = float(self.cfg.b)
        q = float(self.cfg.q)
        rho = float(self.cfg.rho)
        # expected audit sketches per round = rho*q, each 64 bits (int64)
        audit_bits_amortized = (rho * q * 64.0) / max(1.0, float(d))
        return b + audit_bits_amortized

    def communication_expansion_over_32bit(self, *, d: int) -> float:
        return self.communication_bits_per_param(d=d) / 32.0


# ---------------------------
# Example usage (pseudo)
# ---------------------------

if __name__ == "__main__":
    # This main is just a sanity illustration; it won't run without SABackend and actual training.
    cfg = D3FLConfig(b=16, q=8, s=5, tau=0, rho=1.0, quant_scale=1e-4, seed=0)

    print("D3FL* knobs:", cfg)
    print("Example comm bits/param (d=1e6):", D3FL(cfg, sa_backend=None).communication_bits_per_param(d=1_000_000))  # type: ignore
    print("Example comm expansion over 32-bit:", D3FL(cfg, sa_backend=None).communication_expansion_over_32bit(d=1_000_000))  # type: ignore
    print("\nImplement SABackend + your FL runner to use this module.")
