"""PyTorch implementation of the `ntc_encoder_heads_v1` architecture.

Mirrors the normative Rust CPU reference (`crates/ntc-model/src/cpu.rs`).
"""

from ntc_model.config import ARCHITECTURE, Calibration, NtcArchConfig, tiny_config

__all__ = ["ARCHITECTURE", "Calibration", "NtcArchConfig", "tiny_config"]
