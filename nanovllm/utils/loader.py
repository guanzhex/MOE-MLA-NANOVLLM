import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, strict: bool = False):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found under {path}")

    loaded_params = set()
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                for k in packed_modules_mapping:
                    if f".{k}." in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v, 1)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError as exc:
                            raise KeyError(
                                f"checkpoint parameter {weight_name!r} maps to missing "
                                f"runtime parameter {param_name!r}"
                            ) from exc
                        weight_loader = getattr(param, "weight_loader")
                        try:
                            weight_loader(param, loaded_weight, shard_id)
                        except (AssertionError, RuntimeError, ValueError) as exc:
                            raise ValueError(
                                f"failed loading {weight_name} {tuple(loaded_weight.shape)} "
                                f"into {param_name} {tuple(param.shape)} shard={shard_id}"
                            ) from exc
                        loaded_params.add(param_name)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError as exc:
                        raise KeyError(
                            f"checkpoint contains unknown runtime parameter {weight_name!r}"
                        ) from exc
                    if param.shape != loaded_weight.shape:
                        raise ValueError(
                            f"shape mismatch for {weight_name}: checkpoint "
                            f"{tuple(loaded_weight.shape)} != runtime {tuple(param.shape)}"
                        )
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(weight_name)

    if strict:
        missing = [name for name, _ in model.named_parameters() if name not in loaded_params]
        if missing:
            preview = ", ".join(missing[:20])
            suffix = "" if len(missing) <= 20 else f" ... and {len(missing) - 20} more"
            raise KeyError(
                f"checkpoint did not initialize runtime parameters: {preview}{suffix}"
            )
