"""Unlit inference adapter for Tiny3D's shared Warp backend."""
from easydict import EasyDict
from tiny3d_renderer import MeshRenderer as _MeshRenderer, intrinsics_to_projection


class MeshRenderer(_MeshRenderer):
    def render(self, *args, **kwargs):
        return EasyDict(super().render(*args, **kwargs))
