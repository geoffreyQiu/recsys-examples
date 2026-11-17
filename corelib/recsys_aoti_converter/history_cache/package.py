# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Declarative loading of original model packages; no architecture dispatch."""
import importlib
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch


def load_export_json(package, config, batch_size, device):
    """Load a serialized ExportedProgram JSON plus its separate tensor checkpoint.

    PyTorch's verifier remains enabled; parameter, buffer, and constant kinds
    come from the serialized signature.
    """
    from torch._export.serde import serialize as serde
    from torch.utils import _pytree as pytree
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    raw = serde._bytes_to_dataclass(serde.ExportedProgram,
                                  (package / config['graph']).read_bytes())
    state = torch.load(package / config['checkpoint'], mmap=True,
                       weights_only=True, map_location='cpu')
    constants = {}
    for item in raw.graph_module.signature.input_specs:
        if item.type == 'parameter':
            name = item.parameter.parameter_name
            state[name] = torch.nn.Parameter(state[name], requires_grad=False)
        elif item.type == 'tensor_constant':
            name = item.tensor_constant.tensor_constant_name
            constants[name] = state.pop(name)
    schema = json.loads((package / config['schema']).read_text())
    with np.load(package / config['sample']['path']) as archive:
        inputs = tuple(torch.from_numpy(archive[x['name']].copy()) for x in schema['inputs'])
    if batch_size < 1 or any(x.shape[0] < batch_size for x in inputs):
        raise ValueError('Requested batch exceeds the supplied sample')
    inputs = tuple(x[:batch_size].contiguous() for x in inputs)
    ep = serde.ExportedProgramDeserializer().deserialize(raw, state, constants, ((inputs,), {}))
    original = ep.module()
    model = torch.fx.GraphModule(original, original.graph)
    # The serialized program has explicit CPU factory/assertion devices.
    # Relocate those together with weights and inputs before recapturing.
    for node in model.graph.nodes:
        relocate = lambda value: torch.device(device) if isinstance(value, torch.device) else value
        node.args = pytree.tree_map(relocate, node.args)
        node.kwargs = pytree.tree_map(relocate, node.kwargs)
    model.recompile()
    return model.to(device).eval(), tuple(x.to(device) for x in inputs), None, config


def load_package(package, descriptor, batch_size, device="cuda"):
    package, descriptor = Path(package), Path(descriptor)
    config = json.loads(descriptor.read_text())
    if config.get('format') == 'export_json':
        return load_export_json(package, config, batch_size, device)
    source = package / config.get("source", ".")
    # Original packages can use the same Python package name. Load them under
    # separate source namespaces so one process can inspect both safely.
    namespace = '_history_source_' + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        root_module = types.ModuleType(namespace)
        root_module.__path__ = [str(source)]
        root_module.__package__ = namespace
        sys.modules[namespace] = root_module
    module, name = config["class"].split(":")
    factory = getattr(importlib.import_module(namespace + '.' + module), name)
    names_module, names_attr = config["input_names"].split(":")
    names = getattr(importlib.import_module(namespace + '.' + names_module), names_attr)
    model = factory(**json.loads((source / config.get("hyperparameters", "hyperparameters.json")).read_text())).eval()
    model.load_state_dict(torch.load(source / config.get("checkpoint", "model.pt"), weights_only=True, map_location="cpu"), strict=True)
    sample = config["sample"]
    if sample["format"] == "npz":
        with np.load(package / sample["path"]) as archive:
            tensors = {name: torch.from_numpy(archive[name].copy()) for name in names}
    elif sample["format"] == "triton_json":
        dtypes = {"FP32": torch.float32, "INT64": torch.int64, "INT32": torch.int32, "BOOL": torch.bool}
        request = json.loads((package / sample["path"]).read_text())
        tensors = {x["name"]: torch.tensor(x["data"], dtype=dtypes[x["datatype"]]).reshape(x["shape"])
                   for x in request["inputs"]}
    else:
        raise ValueError(f"Unsupported sample format: {sample['format']}")
    if batch_size < 1 or any(tensors[name].shape[0] < batch_size for name in names):
        raise ValueError("Requested batch exceeds the supplied sample")
    inputs = {name: tensors[name][:batch_size].clone().to(device) for name in names}
    expected = None
    if config.get("expected_output"):
        expected = torch.from_numpy(np.load(package / config["expected_output"])[:batch_size]).to(device)
    return model.to(device), inputs, expected, config
