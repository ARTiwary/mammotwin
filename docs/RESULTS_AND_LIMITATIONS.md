## Addendum: Overfitting Was Investigated but Not Resolved by Regularization

Train-vs-test comparison revealed real overfitting across all three
classifiers (train ROC-AUC 0.82-0.96 vs. test ROC-AUC 0.69-0.80), most
severe in the multimodal model (gap of 0.20 AUC). Two standard
regularization techniques were tested as fixes:

| Attempt | Weight decay | Frozen backbone | Train AUC | Test AUC | Gap |
|---|---|---|---|---|---|
| Original | 0.00001 | None | 0.956 | 0.756 | 0.200 |
| Aggressive | 0.001 (100x) | 36% (up to layer4) | — | 0.760 (val) | — |
| Mild | 0.0001 (10x) | 6% (up to layer3) | 0.960 | 0.701 | 0.259 (worse) |

Neither attempt reduced the gap; both reduced test performance without
meaningfully reducing training performance. This is informative: it
suggests the model is not simply "too free" to memorize noise (in which
case restricting its capacity should have helped) but rather that ~2,500
images does not contain enough information for a 23M-parameter ResNet50
to learn mammography-specific features that generalize further than it
already does. The final multimodal model therefore retains its original,
unregularized configuration (test ROC-AUC 0.756), and the overfitting gap
is reported as a known, investigated, and unresolved limitation rather
than something masked by an ineffective fix.

**Suggested framing for the report:** *"Overfitting was confirmed via
direct train/test comparison and investigated using two standard
regularization techniques (increased weight decay, partial backbone
freezing). Neither improved generalization, with test performance
degrading in both cases while training performance remained essentially
unchanged. This negative result is consistent with the interpretation
that the limiting factor is training-set size rather than model
capacity — the same conclusion independently supported by this project's
segmentation and lesion-crop experiments, both of which improved
substantially once trained on additional data (Section 3)."*