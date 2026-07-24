# Phase 6: Pretraining

- [ ] Choose optimizer (AdamW is standard) and hyperparameters
- [ ] Set learning rate schedule (warmup + cosine/linear decay)
- [ ] Set batch size and gradient accumulation strategy
- [ ] Decide total training tokens/steps (compute-optimal scaling)
- [ ] Run small-scale pilot/debug run before full launch
- [ ] Launch full pretraining run
- [ ] Monitor loss curves, gradient norms, and for divergence/instability
- [ ] Periodically evaluate on held-out validation set
- [ ] Save regular checkpoints
