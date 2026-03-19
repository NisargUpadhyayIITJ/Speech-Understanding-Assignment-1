# Question 2 Reproduction

This folder contains a reduced reproduction of the paper *Disentangled Representation Learning for Environment-agnostic Speaker Recognition* and a stronger improved variant motivated by the critique.

## Files

- `train.py`: trains one experiment from a YAML config
- `eval.py`: evaluates one or more saved checkpoints and writes aggregate result tables/plots
- `configs/`: baseline, disentangled, and improved experiment settings
- `results/`: checkpoints, metrics, plots, and comparison tables
- `review.pdf`: critical review of the paper

## Recommended Reproduction Commands

Run from the repository root inside the same virtual environment used for Question 1.

### 1. Train the baseline

```bash
python3 -m q2.train --config q2/configs/baseline.yaml
```

### 2. Train the reduced disentangler reproduction

```bash
python3 -m q2.train --config q2/configs/disentangle.yaml
```

### 3. Train the improved variant

```bash
python3 -m q2.train --config q2/configs/improved.yaml
```

### 4. Evaluate the saved checkpoints

```bash
python3 -m q2.eval \
  --checkpoints \
    q2/results/checkpoints/baseline_smallcnn/best.pt \
    q2/results/checkpoints/disentangle_smallcnn/best.pt \
    q2/results/checkpoints/improved_smallcnn/best.pt \
  --environment-name telephone
```

## Checkpoint-to-Result Mapping

After training, the intended mapping is:

- `q2/results/checkpoints/baseline_smallcnn/best.pt` -> baseline metrics and plots
- `q2/results/checkpoints/disentangle_smallcnn/best.pt` -> reduced reproduction metrics and plots
- `q2/results/checkpoints/improved_smallcnn/best.pt` -> critique-motivated improved model metrics and plots

The aggregate comparison outputs will be written to:

- `q2/results/comparison_table.csv`
- `q2/results/comparison_table.json`
- `q2/results/model_comparison.png`
- `q2/results/result_summary.md`

Best checkpoint selection rule:

- each `best.pt` is chosen by the lowest augmented-condition EER
- augmented top-1 accuracy is used only as a tie-break when EER is identical

## Reduced-Reproduction Note

The original paper is evaluated on VoxCeleb with stronger speaker encoders and environment variation coming from in-the-wild recordings. This reproduction uses a small LibriSpeech subset and synthetic environment transforms so that the core disentangling idea remains testable in a compact assignment setting.
