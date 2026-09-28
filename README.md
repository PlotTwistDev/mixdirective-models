# Mix Directive models

This repo is a mirror of the on-device models that the Mix Directive desktop app downloads the first
time you use lyric auto-alignment. The files are attached to the
[releases](https://github.com/PlotTwistDev/mixdirective-models/releases). The primary copies are on
Hugging Face, and the app only uses this mirror if Hugging Face is unavailable. It checks each file's
SHA-256 before installing it, whichever source the file comes from.

The app's source code is not in this repo. It contains only these redistributed models, their licences
and the scripts that produced them.

| Release file | SHA-256 | Primary copy | Licence |
|---|---|---|---|
| `wav2vec2-base-960h.int8.onnx` | `02bcb2e690107934e4636ff883a2623f2f4a186530d42594296df87711789202` | [MixDirective/wav2vec2-base-960h-onnx](https://huggingface.co/MixDirective/wav2vec2-base-960h-onnx) | Apache-2.0 |
| `open-unmix-umxhq-vocals.onnx` | `f27742bb52b24cb039614dc76d0764e279df3eb2a9d6daa82c6daee3369ec747` | [MixDirective/open-unmix-umxhq-vocals-onnx](https://huggingface.co/MixDirective/open-unmix-umxhq-vocals-onnx) | MIT |

## wav2vec2-base-960h.int8.onnx

This is a modified version of [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
(revision `22aad52d`) by Meta AI, licensed Apache-2.0. It was converted to ONNX, and its MatMul layers were
dynamically quantised to int8. It is used for CTC forced alignment of English lyrics.
Licence: [`licenses/Apache-2.0-wav2vec2-base-960h.txt`](licenses/Apache-2.0-wav2vec2-base-960h.txt).
Produced by [`scripts/export_wav2vec2_ctc_onnx.py`](scripts/export_wav2vec2_ctc_onnx.py).

> Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*
> [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)

## open-unmix-umxhq-vocals.onnx

This is a modified version of the vocals network from [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch)
UMX-HQ (weights: [Zenodo 3370489](https://doi.org/10.5281/zenodo.3370489)), © 2019 Inria, licensed MIT.
Only the vocals network is included, converted to ONNX with a dynamic frame axis. The STFT and iSTFT
steps are left out. It is used to isolate vocals from a full mix before alignment.
Licence: [`licenses/MIT-open-unmix.txt`](licenses/MIT-open-unmix.txt).
Produced by [`scripts/export_umxhq_vocals_onnx.py`](scripts/export_umxhq_vocals_onnx.py).

> Stöter et al. (2019). *Open-Unmix – A Reference Implementation for Music Source Separation.* JOSS 4(41), 1667.
> https://doi.org/10.21105/joss.01667

Full model cards (inputs, outputs and evaluation) are on the Hugging Face pages linked above.
