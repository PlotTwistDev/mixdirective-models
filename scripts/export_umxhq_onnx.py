"""Export Open-Unmix UMX-HQ target networks (MIT weights) to ONNX.

Developer tool, not shipped; the same recipe as export_umxhq_vocals_onnx.py
for any of the four targets (vocals, drums, bass, other), used by Music
Rebalance (app/core/music_rebalance.py). Needs torch + openunmix + onnx +
onnxruntime. Weights: https://zenodo.org/records/3370489 (UMX-HQ, MIT; the
newer umxl weights are non-commercial and must not be used).

    python3 scripts/export_umxhq_onnx.py out_dir drums bass other

Writes <out_dir>/umxhq-<target>/model.onnx and manifest.json (SHA-256, and
the ONNX-vs-PyTorch parity) per target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_umxhq_vocals_onnx import _dynamic_core, _sha256  # noqa: E402

TARGETS = ("vocals", "drums", "bass", "other")
WEIGHTS = "https://zenodo.org/records/3370489/files/{}"


def export(target: str, out_dir: Path) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch
    from openunmix import umxhq_spec

    folder = out_dir / f"umxhq-{target}"
    folder.mkdir(parents=True, exist_ok=True)
    model = umxhq_spec(targets=[target], pretrained=True)[target].eval()
    path = folder / "model.onnx"
    torch.onnx.export(
        _dynamic_core(model), (torch.rand(1, 2, 2049, 200),), str(path),
        input_names=["magnitude"], output_names=[f"{target}_magnitude"],
        dynamic_axes={"magnitude": {3: "frames"}, f"{target}_magnitude": {3: "frames"}},
        opset_version=17, dynamo=False,
    )
    rng = np.random.default_rng(0)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    parity = {}
    for frames in (57, 1300):
        magnitude = np.abs(rng.standard_normal((1, 2, 2049, frames))).astype(np.float32)
        with torch.inference_mode():
            reference = model(torch.from_numpy(magnitude)).numpy()
        got = session.run(None, {"magnitude": magnitude})[0]
        parity[str(frames)] = {
            "max_abs_diff": float(np.max(np.abs(got - reference))),
            "max_rel_diff": float(np.max(np.abs(got - reference)) / max(1e-9, float(np.max(np.abs(reference))))),
        }
    manifest = {
        "model": f"Open-Unmix UMX-HQ ({target})",
        "weights": WEIGHTS.format({"vocals": "vocals-b62c91ce.pth", "drums": "drums-9619578f.pth",
                                   "bass": "bass-8d85a5bd.pth", "other": "other-b52fbbf7.pth"}[target]),
        "license": "MIT (© 2019 Inria)",
        "stft": {"sample_rate": 44100, "n_fft": 4096, "hop": 1024, "window": "hann (periodic)", "center": True},
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}},
        "parity": parity,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str]) -> int:
    out = Path(argv[0] if argv else "umxhq-onnx")
    targets = argv[1:] or list(TARGETS)
    for target in targets:
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        print(json.dumps(export(target, out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
