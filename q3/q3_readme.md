# Question 3 Reproduction Guide

This folder contains the full reduced implementation for the ethical-auditing task using the English split of Common Voice hosted at `fsicoli/common_voice_17_0`.

## Files

- `audit.py`: metadata audit for documentation debt and representation imbalance
- `privacymodule.py`: PyTorch biometric obfuscation module
- `pp_demo.py`: privacy-preserving transformation demo with ASR retention and proxy quality checks
- `train_fair.py`: fairness-aware ASR experiment using a frozen `facebook/wav2vec2-base-960h` backbone and a trainable logit adapter
- `evaluation_scripts/proxy_metrics.py`: SNR and log-spectral-distance acceptability proxies
- `examples/`: original and obfuscated WAV pairs used in the demo
- `audit_plots.pdf`: audit visualizations
- `q3_report.pdf`: final report

## Commands

Run from the repository root with the assignment virtual environment active.

```bash
python3 -m q3.audit --split validated --max-examples 1200 --refresh
python3 -m q3.pp_demo --subset-size 8 --refresh
python3 -m q3.train_fair --refresh-data
```

Regenerate the report PDF:

```bash
pandoc --resource-path=q3 q3/q3_report.md -o q3/q3_report.docx
libreoffice --headless --convert-to pdf q3/q3_report.docx --outdir q3
```

## Output Mapping

- Audit summary: `q3/results/audit_summary.json`
- Audit plots: `q3/audit_plots.pdf`
- Privacy demo summary: `q3/results/privacy_demo_summary.json`
- Privacy demo table: `q3/results/privacy_demo_table.csv`
- Privacy demo plot: `q3/results/privacy_demo_metrics.png`
- Fairness comparison JSON: `q3/results/fairness/comparison.json`
- Fairness comparison table: `q3/results/fairness/comparison_table.csv`
- Fairness plot: `q3/results/fairness/fairness_comparison.png`
- Fairness checkpoints:
  - `q3/results/fairness/checkpoints/adapter_ctc.pt`
  - `q3/results/fairness/checkpoints/adapter_fair.pt`

## Notes

- The audit uses reservoir sampling over `validated.tsv` so the metadata analysis is not biased toward the start of the file.
- The privacy demo uses a lightweight spectral obfuscator, not a neural voice-conversion model, so it should be described as a reduced privacy-preserving transformation.
- The fairness experiment is intentionally lightweight: a frozen pretrained ASR model plus a trainable adapter and fairness penalty. In this reduced setup, the fairness term improves validation parity but does not materially improve held-out test parity.
