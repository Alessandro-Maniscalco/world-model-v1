"""Tests for the vendored Wan VACE backbone import surface."""

from __future__ import annotations


def test_vendor_wan_vace_backbone_imports() -> None:
    """Expose the local vendored Wan VACE backbone symbol."""
    from world_model.vendor.wan import WanVACETransformer3DModel

    assert WanVACETransformer3DModel is not None
