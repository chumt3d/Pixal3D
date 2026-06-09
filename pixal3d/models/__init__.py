import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

_OPTIONAL_REGENERATED_BUFFER_KEYS = frozenset({"rope_phases"})
_SAFETENSORS_TORCH_DTYPES = {
    "BOOL": ("bool", 1),
    "U8": ("uint8", 1),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "I32": ("int32", 4),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I64": ("int64", 8),
}

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _safetensors_element_count(shape):
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError(f"Invalid safetensors shape dimension: {shape!r}")
        total *= dim
    return total


def _load_safetensors_without_regenerated_buffers(model_file: str):
    import json
    import struct
    import torch

    state_dict = {}
    with open(model_file, "rb") as f:
        raw_header_size = f.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"Missing safetensors header size: {model_file}")
        header_size = struct.unpack("<Q", raw_header_size)[0]
        header = json.loads(f.read(header_size).rstrip())
        data_start = 8 + header_size
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_name = info["dtype"]
            if dtype_name == "C64" and name in _OPTIONAL_REGENERATED_BUFFER_KEYS:
                continue
            dtype_record = _SAFETENSORS_TORCH_DTYPES.get(dtype_name)
            if dtype_record is None:
                raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} for tensor {name!r}")
            torch_dtype_name, byte_size = dtype_record
            torch_dtype = getattr(torch, torch_dtype_name)
            shape = info["shape"]
            offsets = info["data_offsets"]
            if (
                not isinstance(shape, list)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"Invalid safetensors metadata for tensor {name!r}")
            expected_bytes = _safetensors_element_count(shape) * byte_size
            if offsets[1] - offsets[0] != expected_bytes:
                raise ValueError(f"Invalid safetensors byte range for tensor {name!r}")
            f.seek(data_start + offsets[0])
            raw_tensor = bytearray(f.read(expected_bytes))
            if len(raw_tensor) != expected_bytes:
                raise ValueError(f"Truncated safetensors payload for tensor {name!r}")
            tensor = torch.frombuffer(raw_tensor, dtype=torch_dtype)
            state_dict[name] = tensor.reshape(shape).clone()
    return state_dict


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    try:
        model_ckpt = load_file(model_file)
    except Exception:
        model_ckpt = _load_safetensors_without_regenerated_buffers(model_file)
    model.load_state_dict(model_ckpt, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
