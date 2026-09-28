"""Export Open-Unmix UMX-HQ's vocals model (MIT weights) to ONNX.

Developer tool, not shipped. Needs torch + openunmix + onnx + onnxruntime:

    uv venv /tmp/umx && uv pip install --python /tmp/umx torch openunmix onnx onnxruntime
    /tmp/umx/bin/python scripts/export_umxhq_vocals_onnx.py out_dir

Only the network is exported: magnitude spectrogram in, estimated vocal
magnitude out. The STFT/iSTFT around it (n_fft 4096, hop 1024, periodic Hann,
centred with reflect padding, 44.1 kHz stereo) run in numpy in the app
(app/core/onnx_vocal_separator.py), so onnxruntime needs no STFT operators.

Weights: https://zenodo.org/records/3370489 ("Open-Unmix-Pytorch UMX-HQ",
MIT licence). Writes model.onnx, manifest.json (SHA-256) and checks the ONNX
output against PyTorch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

WEIGHTS_URL = "https://zenodo.org/records/3370489/files/vocals-b62c91ce.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dynamic_core(model):
    """OpenUnmix.forward with frame-count-free reshapes.

    The original reads ``x.data.shape`` into Python ints, which tracing bakes
    into the graph as the example's frame count. Same layers and order, but
    batch 1 / stereo fixed and frames left as -1.
    """
    import torch
    import torch.nn.functional as F

    class Core(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, magnitude):
            m = self.m
            x = magnitude.permute(3, 0, 1, 2)  # frames, 1, 2, bins
            mix = x
            x = x[..., : m.nb_bins]
            x = (x + m.input_mean) * m.input_scale
            x = m.bn1(m.fc1(x.reshape(-1, 2 * m.nb_bins)))
            x = torch.tanh(x.reshape(-1, 1, m.hidden_size))
            x = torch.cat([x, m.lstm(x)[0]], -1)
            x = F.relu(m.bn2(m.fc2(x.reshape(-1, 2 * m.hidden_size))))
            x = m.bn3(m.fc3(x)).reshape(-1, 1, 2, m.nb_output_bins)
            x = x * m.output_scale + m.output_mean
            return (F.relu(x) * mix).permute(1, 2, 3, 0)

    return Core(model).eval()


def main(out_dir: str) -> int:
    import numpy as np
    import onnxruntime as ort
    import torch
    from openunmix import umxhq_spec

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = umxhq_spec(targets=["vocals"], pretrained=True)["vocals"].eval()

    path = out / "model.onnx"
    dummy = torch.rand(1, 2, 2049, 200)
    torch.onnx.export(
        _dynamic_core(model),
        (dummy,),
        str(path),
        input_names=["magnitude"],
        output_names=["vocals_magnitude"],
        dynamic_axes={"magnitude": {3: "frames"}, "vocals_magnitude": {3: "frames"}},
        opset_version=17,
        dynamo=False,
    )

    rng = np.random.default_rng(0)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    parity = {}
    for frames in (57, 1300):
        mag = np.abs(rng.standard_normal((1, 2, 2049, frames))).astype(np.float32)
        with torch.inference_mode():
            reference = model(torch.from_numpy(mag)).numpy()
        got = session.run(None, {"magnitude": mag})[0]
        parity[str(frames)] = {
            "max_abs_diff": float(np.max(np.abs(got - reference))),
            "max_rel_diff": float(np.max(np.abs(got - reference)) / max(1e-9, float(np.max(np.abs(reference))))),
        }
    manifest = {
        "model": "Open-Unmix UMX-HQ (vocals)",
        "weights": WEIGHTS_URL,
        "license": "MIT",
        "stft": {"sample_rate": 44100, "n_fft": 4096, "hop": 1024, "window": "hann (periodic)", "center": True},
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}},
        "parity": parity,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "umxhq-vocals-onnx"))
