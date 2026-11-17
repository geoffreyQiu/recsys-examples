# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Precision policy shared by export, runtime, validation, and benchmarks."""
import os
import torch

PRECISIONS = ('fp32', 'bf16')
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}
# BF16 has seven mantissa bits. Report discrete decisions separately: a
# numerical tolerance is not evidence of identical greedy/ranking behavior.
TOLERANCES = {'fp32': {'rtol': 1e-4, 'atol': 1e-5},
              'bf16': {'rtol': 2e-2, 'atol': 2e-3}}


def precision_name(value):
    for name, dtype in DTYPES.items():
        if value in (name, dtype, str(dtype), str(dtype).removeprefix('torch.')):
            return name
    raise ValueError(f'Unsupported precision {value!r}; choose fp32 or bf16')


def configure_math(precision='fp32', tf32=False):
    """Set the same explicit FP32 math policy before capture, compile and run.

    TF32 changes eligible FP32 kernel arithmetic, never tensor/cache dtypes.
    These backend settings are process-wide; use one policy per serving process.
    """
    precision = precision_name(precision)
    if not isinstance(tf32, bool):
        raise ValueError('tf32 must be a boolean')
    if tf32 and precision != 'fp32':
        raise ValueError('TF32 is an option for --precision fp32 only')
    if tf32 and os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
        raise ValueError('NVIDIA_TF32_OVERRIDE=0 conflicts with enabling TF32')
    mode = 'tf32' if tf32 else 'ieee'
    # This build's exporter still saves/restores legacy cuDNN flags, while
    # Inductor reads fp32_precision. Initialize the legacy flags first, then
    # give both APIs the same policy; new-only cuDNN setup breaks export.
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.fp32_precision = mode
    torch.backends.cudnn.fp32_precision = mode
    return {'tf32': tf32,
            'cuda_matmul_fp32_precision': torch.backends.cuda.matmul.fp32_precision,
            'cudnn_fp32_precision': torch.backends.cudnn.fp32_precision}


def check_math_policy(tf32=False):
    mode = 'tf32' if tf32 else 'ieee'
    if (torch.backends.cuda.matmul.fp32_precision != mode
            or torch.backends.cudnn.fp32_precision != mode):
        raise RuntimeError('FP32 math policy changed after loading the bundle; '
                           'restore its policy or use a separate serving process')


def tensor_bytes(tensor):
    # NumPy does not expose torch.bfloat16. Hash storage bytes without a
    # lossy conversion, including scalars and noncontiguous inputs.
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def artifact_dir(root, milestone, batch, precision='fp32'):
    suffix = '' if precision_name(precision) == 'fp32' else '_bf16'
    return root / 'artifacts' / f'{milestone}_b{batch}{suffix}'


class PrecisionModel(torch.nn.Module):
    """Keep original inputs/weights and encode CUDA autocast in the export."""
    def __init__(self, model, precision='fp32'):
        super().__init__()
        self.inner = model
        self.precision = precision_name(precision)

    def forward(self, *args, **kwargs):
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=self.precision == 'bf16'):
            return self.inner(*args, **kwargs)


def export_model(model, example_args, precision='fp32'):
    precision = precision_name(precision)
    if precision == 'fp32':
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            return torch.export.export(model.eval(), example_args, strict=False)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is not supported on the selected GPU')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        exported = torch.export.export(PrecisionModel(model.eval(), precision),
                                       example_args, strict=False)
        # Exporting only inside an external autocast context loses that policy
        # on replay. Lower the captured context into explicit ATen casts while
        # retaining linears for structural matching. AOTI needs no ambient AMP.
        return exported.run_decompositions({})


def compare(actual, expected, *, tolerance=None):
    actual, expected = actual.detach().float(), expected.detach().float()
    diff = (actual - expected).abs()
    order, reference_order = actual.argsort(dim=-1), expected.argsort(dim=-1)
    result = {'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
            'rmse': diff.square().mean().sqrt().item(),
            'max_rel': (diff / expected.abs().clamp_min(1e-6)).max().item(),
            'finite': bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            'argsort_equal': bool(torch.equal(order, reference_order)),
            'argsort_positions_equal': int((order == reference_order).sum().item()),
            'argsort_positions_total': order.numel()}
    if tolerance is not None:
        close = torch.isclose(actual, expected, **tolerance)
        result.update({'numeric_within_tolerance': bool(close.all()),
                       'numeric_mismatches': int((~close).sum()),
                       'numeric_elements': close.numel(), 'tolerance': tolerance})
    return result


def compile_options(precision='fp32'):
    options = {'max_autotune': False, 'compile_threads': 4}
    if precision_name(precision) == 'bf16':
        # Greedy decoders amplify tiny differences. Preserve eager BF16
        # rounding at intermediate casts instead of silently eliding it.
        options['emulate_precision_casts'] = True
    return options
