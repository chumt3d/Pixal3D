"""Unlit inference adapter for Tiny3D's shared Warp backend."""
from easydict import EasyDict
from tiny3d_renderer import MeshRenderer as _MeshRenderer, intrinsics_to_projection


class MeshRenderer(_MeshRenderer):
    def __init__(self, rendering_options=None, device="cuda"):
        super().__init__(rendering_options, device)
        self.rendering_options = EasyDict(vars(self.rendering_options))

    def render(self, *args, **kwargs):
        return EasyDict(super().render(*args, **kwargs))
