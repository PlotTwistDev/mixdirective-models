"""Export facebook/wav2vec2-base-960h (Apache-2.0) to ONNX for the lyric aligner.

Developer tool, not shipped. Needs torch + transformers + onnx, which the app
itself never imports:

    uv venv /tmp/w2v2 && uv pip install --python /tmp/w2v2 torch transformers onnx onnxruntime
    /tmp/w2v2/bin/python scripts/export_wav2vec2_ctc_onnx.py out_dir

Writes ``model.onnx`` (fp32), ``model.int8.onnx`` (dynamic int8 quantised),
``vocab.json`` and ``manifest.json`` with SHA-256 digests, then checks both
ONNX files against the PyTorch logits. The app pins the digest of whichever
file it downloads (app/core/lyrics_alignment.py, _W2V2_MODEL).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

MODEL = "facebook/wav2vec2-base-960h"
REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(out_dir: str) -> int:
    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REVISION).eval()
    processor = Wav2Vec2Processor.from_pretrained(MODEL, revision=REVISION)

    fp32 = out / "model.onnx"
    dummy = torch.zeros(1, 16000 * 5)
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
        opset_version=17,
        dynamo=False,
    )
    int8 = out / "model.int8.onnx"
    # MatMul only: onnxruntime's CPU provider has no ConvInteger kernel, and
    # the transformer's MatMuls hold nearly all of the weights anyway.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    (out / "vocab.json").write_text(json.dumps(processor.tokenizer.get_vocab(), sort_keys=True))

    # Parity check on noise shaped like normalised speech input, two lengths.
    rng = np.random.default_rng(0)
    report = {}
    for seconds in (3.7, 15.0):
        wave = rng.standard_normal(int(16000 * seconds)).astype(np.float32)
        with torch.inference_mode():
            reference = model(torch.from_numpy(wave)[None]).logits[0].numpy()
        for path in (fp32, int8):
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            logits = session.run(None, {"input_values": wave[None]})[0][0]
            agree = float(np.mean(logits.argmax(-1) == reference.argmax(-1)))
            report[f"{path.name}@{seconds}s"] = {
                "frames": int(logits.shape[0]),
                "max_abs_diff": float(np.max(np.abs(logits - reference))),
                "argmax_agreement": agree,
            }
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "license": "Apache-2.0",
        "files": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in (fp32, int8)},
        "parity": report,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "w2v2-onnx"))
