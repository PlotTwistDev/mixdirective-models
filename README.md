# Mix Directive models

This repo is a mirror of the on-device models that the Mix Directive desktop app downloads the first
time you use a feature that needs one: lyric auto-alignment, vocal separation, Music Rebalance, Reduce
Bleed's Model mode and the Learned method of Reduce Reverb. The files are attached to the
[`models-v1` release](https://github.com/PlotTwistDev/mixdirective-models/releases/tag/models-v1). The
primary copies are on Hugging Face, and the app only uses this mirror if Hugging Face is unavailable. It
checks each file's SHA-256 before installing it, whichever source the file comes from.

The app's source code is not in this repo. It contains only these models, their licences and the
scripts that produced the redistributed ones.

| Release file | SHA-256 | Primary copy | Licence |
|---|---|---|---|
| `wav2vec2-base-960h.int8.onnx` | `02bcb2e690107934e4636ff883a2623f2f4a186530d42594296df87711789202` | [MixDirective/wav2vec2-base-960h-onnx](https://huggingface.co/MixDirective/wav2vec2-base-960h-onnx) | Apache-2.0 |
| `open-unmix-umxhq-vocals.onnx` | `f27742bb52b24cb039614dc76d0764e279df3eb2a9d6daa82c6daee3369ec747` | [MixDirective/open-unmix-umxhq-vocals-onnx](https://huggingface.co/MixDirective/open-unmix-umxhq-vocals-onnx) | MIT |
| `open-unmix-umxhq-drums.onnx` | `471628e1e48138a634f962a89f622929980cf5485ed0a4e6e648e9dfcd1145d9` | [MixDirective/open-unmix-umxhq-drums-onnx](https://huggingface.co/MixDirective/open-unmix-umxhq-drums-onnx) | MIT |
| `open-unmix-umxhq-bass.onnx` | `8931033f04666fdcf300d610579351d74275a6c55dd4346009290c844033726f` | [MixDirective/open-unmix-umxhq-bass-onnx](https://huggingface.co/MixDirective/open-unmix-umxhq-bass-onnx) | MIT |
| `open-unmix-umxhq-other.onnx` | `fe62c1d8fb8326c7e68540a675dfa186a2b131f763fb41050ea5e9a9859eb76e` | [MixDirective/open-unmix-umxhq-other-onnx](https://huggingface.co/MixDirective/open-unmix-umxhq-other-onnx) | MIT |
| `mix-directive-dereverb.onnx` | `db0d2df647e946751ca80613f3909d2bc3aa83bafc6566b2d604834911bbea37` | [MixDirective/mix-directive-dereverb-onnx](https://huggingface.co/MixDirective/mix-directive-dereverb-onnx) | Proprietary |

## wav2vec2-base-960h.int8.onnx

This is a modified version of [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
(revision `22aad52d`) by Meta AI, licensed Apache-2.0. It was converted to ONNX, and its MatMul layers were
dynamically quantised to int8. It is used for CTC forced alignment of English lyrics.
Licence: [`licenses/Apache-2.0-wav2vec2-base-960h.txt`](licenses/Apache-2.0-wav2vec2-base-960h.txt).
Produced by [`scripts/export_wav2vec2_ctc_onnx.py`](scripts/export_wav2vec2_ctc_onnx.py).

> Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*
> [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)

## open-unmix-umxhq-{vocals,drums,bass,other}.onnx

These are modified versions of the four target networks from
[Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) UMX-HQ (weights:
[Zenodo 3370489](https://doi.org/10.5281/zenodo.3370489)), © 2019 Inria, licensed MIT. Each file holds
one target's network, converted to ONNX (opset 17) with a dynamic frame axis. The weights are unchanged
(float32). The STFT, iSTFT and Wiener filtering are left out. The vocals network isolates vocals from a
full mix before lyric alignment; all four together drive Music Rebalance and Reduce Bleed's Model mode.
The newer `umxl` weights are licensed for non-commercial use only and are not used.
Licence: [`licenses/MIT-open-unmix.txt`](licenses/MIT-open-unmix.txt).
Produced by [`scripts/export_umxhq_vocals_onnx.py`](scripts/export_umxhq_vocals_onnx.py) (vocals) and
[`scripts/export_umxhq_onnx.py`](scripts/export_umxhq_onnx.py) (drums, bass, other).

> Stöter et al. (2019). *Open-Unmix – A Reference Implementation for Music Source Separation.* JOSS 4(41), 1667.
> https://doi.org/10.21105/joss.01667

## mix-directive-dereverb.onnx

A dereverberation mask network trained by Mix Directive, used for the Learned method of Reduce Reverb.
It is **not** open source: the weights are © 2026 Mix Directive, all rights reserved, and are published
here only so the app can download them. No other use, copying, modification or redistribution is
permitted without written permission.
Licence: [`licenses/Proprietary-mix-directive-dereverb.txt`](licenses/Proprietary-mix-directive-dereverb.txt).

It was trained on dry recordings convolved with generated room and plug-in-style reverbs. The dry
recordings came from the following datasets, both licensed CC BY 4.0, and from material Mix Directive is
licensed to use. No recordings from these datasets are distributed here.

> Wilkins, J., Seetharaman, P., Wahl, A., & Pardo, B. (2018). *VocalSet: A Singing Voice Dataset.* ISMIR.
> https://zenodo.org/record/1442513

> Yamagishi, J., Veaux, C., & MacDonald, K. (2019). *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR
> Voice Cloning Toolkit (version 0.92).* University of Edinburgh. https://doi.org/10.7488/ds/2645

Full model cards (inputs, outputs and evaluation) are on the Hugging Face pages linked above.
