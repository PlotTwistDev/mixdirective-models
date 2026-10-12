# Mix Directive models

This repo is a mirror of the on-device models that the Mix Directive desktop app downloads the first
time you use lyric auto-alignment or one of the Spectral Editor's model-based tools. The files are attached to the
[releases](https://github.com/PlotTwistDev/mixdirective-models/releases). The primary copies are on
Hugging Face, and the app only uses this mirror if Hugging Face is unavailable. It checks each file's
SHA-256 before installing it, whichever source the file comes from.

The app's source code is not in this repo. It contains only these redistributed models, their licences
and the scripts that produced them.

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

## open-unmix-umxhq-vocals.onnx

This is a modified version of the vocals network from [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch)
UMX-HQ (weights: [Zenodo 3370489](https://doi.org/10.5281/zenodo.3370489)), © 2019 Inria, licensed MIT.
Only the vocals network is included, converted to ONNX with a dynamic frame axis. The STFT and iSTFT
steps are left out. It is used to isolate vocals from a full mix before alignment.
Licence: [`licenses/MIT-open-unmix.txt`](licenses/MIT-open-unmix.txt).
Produced by [`scripts/export_umxhq_vocals_onnx.py`](scripts/export_umxhq_vocals_onnx.py).

> Stöter et al. (2019). *Open-Unmix – A Reference Implementation for Music Source Separation.* JOSS 4(41), 1667.
> https://doi.org/10.21105/joss.01667

## open-unmix-umxhq-drums.onnx, open-unmix-umxhq-bass.onnx, open-unmix-umxhq-other.onnx

These are modified versions of the drums, bass and other networks from
[Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) UMX-HQ (weights:
[Zenodo 3370489](https://doi.org/10.5281/zenodo.3370489)), © 2019 Inria, licensed MIT. Each file holds one
network, converted to ONNX with a dynamic frame axis. The STFT and iSTFT steps are left out. With the vocals
network they are used by Music Rebalance and by Reduce Bleed's Model mode in the Spectral Editor. The newer
non-commercial `umxl` weights are not used.
Licence: [`licenses/MIT-open-unmix.txt`](licenses/MIT-open-unmix.txt).
Produced by [`scripts/export_umxhq_onnx.py`](scripts/export_umxhq_onnx.py).

## mix-directive-dereverb.onnx

A dereverberation mask network trained by Mix Directive, used by Reduce Reverb's Learned method in the
Spectral Editor. It is proprietary, all rights reserved, and published only so the app can download it.
It was trained on generated reverbs applied to dry recordings from VocalSet (Wilkins et al., 2018, CC BY 4.0,
https://zenodo.org/record/1442513) and the CSTR VCTK Corpus 0.92 (Yamagishi, Veaux & MacDonald, University
of Edinburgh, 2019, CC BY 4.0, https://doi.org/10.7488/ds/2645). No recordings from those datasets are included.
Licence: [`licenses/Proprietary-mix-directive-dereverb.txt`](licenses/Proprietary-mix-directive-dereverb.txt).

Full model cards (inputs, outputs and evaluation) are on the Hugging Face pages linked above.
