# Speech Understanding Assignment 1

This repository is now organized by question so the remaining assignment work can live alongside Question 1 cleanly.

## Layout

- `q1/`: complete implementation, data, outputs, and report for Question 1
- `q2/`: paper review, reduced reproduction code, configs, checkpoints, tables, and plots for Question 2
- `q3/`: ethical audit, privacy-preserving demo, fairness-aware ASR experiment, examples, and report for Question 3
- `requirements.txt`: shared Python dependencies for the assignment

## Working With Question 1

Use the detailed Question 1 guide here:

- `q1/README.md`

Common commands from the repository root:

```bash
python3 q1/prepare_librispeech_subset.py --subset clean --split test --num-samples 3 --auto-phone-sequence
python3 q1/run_q1_pipeline.py --manifest-path q1/data/librispeech_subset/manifest.csv --alignment-source transcript --model-name facebook/wav2vec2-base-960h
pandoc --resource-path=q1 q1/q1_report.md -o q1/q1_report.pdf
```

## Working With Question 2

Use the detailed Question 2 guide here:

- `q2/q2_readme.md`

Common commands from the repository root:

```bash
python3 -m q2.train --config q2/configs/baseline.yaml
python3 -m q2.train --config q2/configs/disentangle.yaml
python3 -m q2.train --config q2/configs/improved.yaml
python3 -m q2.eval --checkpoints q2/results/checkpoints/baseline_smallcnn/best.pt q2/results/checkpoints/disentangle_smallcnn/best.pt q2/results/checkpoints/improved_smallcnn/best.pt --environment-name telephone
pandoc --resource-path=q2 q2/review.md -o q2/review.pdf
```

## Working With Question 3

Use the detailed Question 3 guide here:

- `q3/q3_readme.md`

Common commands from the repository root:

```bash
python3 -m q3.audit --split validated --max-examples 1200 --refresh
python3 -m q3.pp_demo --subset-size 8 --refresh
python3 -m q3.train_fair --refresh-data
pandoc --resource-path=q3 q3/q3_report.md -o q3/q3_report.docx
libreoffice --headless --convert-to pdf q3/q3_report.docx --outdir q3
```
