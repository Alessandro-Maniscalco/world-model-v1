"""Vendored Wan backbone modules for local world-model adaptation."""

from world_model.vendor.wan.transformer_wan import WanTransformer3DModel
from world_model.vendor.wan.transformer_wan_vace import WanVACETransformer3DModel

__all__ = ["WanTransformer3DModel", "WanVACETransformer3DModel"]
