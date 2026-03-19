# Question 2 Result Summary

Primary evaluation environment: `telephone`

| Model | Clean Top-1 | Telephone Top-1 | Clean EER | Telephone EER |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`baseline_smallcnn`) | 0.4375 | 0.0625 | 0.2545 | 0.3705 |
| Reduced reproduction (`disentangle_smallcnn`) | 0.3750 | 0.2500 | 0.2991 | 0.3259 |
| Improved variant (`improved_smallcnn`) | 0.4375 | 0.1875 | 0.2411 | 0.3214 |

Best-checkpoint selection rule: lowest augmented-condition EER, with augmented top-1 accuracy used only as a tie-break.

Key takeaways:

- The paper-inspired reduced reproduction improved the mismatch EER over the baseline.
- The proposed speaker-consistency improvement reduced mismatch EER further.
- The disentangled model achieved the highest mismatch-condition top-1 accuracy, while the improved model achieved the best overall EER trade-off.
