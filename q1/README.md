# Question 1

This directory contains the full implementation for Question 1:

- manual MFCC / cepstrum extraction without `librosa.feature.mfcc`
- spectral leakage and SNR comparison across rectangular, Hamming, and Hanning windows
- cepstrum-driven voiced / unvoiced boundary detection
- Hugging Face CTC-based phonetic mapping with boundary RMSE calculation

## Layout

- `mfcc_manual.py`: handcrafted MFCC and cepstrum pipeline
- `leakage_snr.py`: spectral leakage / SNR analysis and plotting
- `voiced_unvoiced.py`: voiced-unvoiced boundary detection plus visualization
- `phonetic_mapping.py`: Hugging Face alignment, phone mapping, and RMSE computation
- `prepare_librispeech_subset.py`: downloads a small `openslr/librispeech_asr` subset and builds a manifest
- `run_q1_pipeline.py`: runs all four Question 1 stages for every sample in the manifest
- `speech_utils/`: shared DSP and CTC alignment helpers
- `data/`: Question 1 data and manifest files
- `outputs/`: generated outputs, logs, and summaries
- `q1_report.md`: report source
- `q1_report.pdf`: generated report

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare a Small LibriSpeech Subset

```bash
python3 q1/prepare_librispeech_subset.py \
  --subset clean \
  --split test \
  --num-samples 3 \
  --auto-phone-sequence
```

This writes:

- local WAV files under `q1/data/librispeech_subset/audio/`
- a manifest CSV with transcript, speaker, chapter, sample ids, and optional phone sequences
- a JSON summary of the exported clips

### 2. Run the Full Question 1 Pipeline

```bash
python3 q1/run_q1_pipeline.py \
  --manifest-path q1/data/librispeech_subset/manifest.csv \
  --alignment-source transcript \
  --model-name facebook/wav2vec2-base-960h
```

This writes:

- stage outputs under `q1/outputs/`
- per-sample logs under `q1/outputs/pipeline_logs/`
- aggregate summaries in `q1/outputs/pipeline_summary.json` and `q1/outputs/pipeline_summary.csv`

### 3. Run Individual Stages

```bash
python3 q1/mfcc_manual.py \
  --audio-path q1/data/librispeech_subset/audio/REPLACE.wav
```

```bash
python3 q1/leakage_snr.py \
  --audio-path q1/data/librispeech_subset/audio/REPLACE.wav
```

```bash
python3 q1/voiced_unvoiced.py \
  --audio-path q1/data/librispeech_subset/audio/REPLACE.wav
```

```bash
python3 q1/phonetic_mapping.py \
  --audio-path q1/data/librispeech_subset/audio/REPLACE.wav \
  --segments-json q1/outputs/voiced_unvoiced/REPLACE_voiced_unvoiced.json \
  --model-name facebook/wav2vec2-base-960h \
  --transcript "REPLACE WITH LIBRISPEECH TRANSCRIPT"
```

## Report

Regenerate the report PDF from the repository root:

```bash
pandoc --resource-path=q1 q1/q1_report.md -o q1/q1_report.pdf
```
