# Plan: Implement the 6 NTK-Theory Practical Modifications in YDF

Source: `ntk_doc.md` Section 5. Six independent ideas, each maps to a different
lever in the training pipeline (split criterion, learning rate, capacity,
feature importance, stopping). All can be implemented as opt-in hyperparameters
without breaking existing behavior.

---

## Cross-cutting design decisions

- **Scope**: Squared-loss / regression-with-hessian first (matches the paper's
  derivation). Multi-class & log-loss extensions are out of scope of this plan
  (paper itself defers them).
- **SNR vs `min_examples`**: when `SplitScore=SNR`, both checks compose —
  a split must pass `min_examples` AND the SNR threshold.
- **Adaptive-η in multi-output rounds**: per-tree η (Option A). Each tree in a
  round gets its own η computed from its own leaves.
- **CapacityPolicy default**: `WARN` — log a one-line warning when
  `T·(2^L − 1) > n_eff` so users discover the bound, but don't change results.
- **Backwards compatibility**: every modification is gated behind a new proto
  field with a default that reproduces today's behavior.
- **Where new options live**: shared knobs (split criterion, capacity check)
  go on `decision_tree.proto::DecisionTreeTrainingConfig` so RF and GBT both
  pick them up. GBT-only knobs (adaptive `eta`, MP-stopping) go on
  `gradient_boosted_trees.proto::GradientBoostedTreesTrainingConfig`.
- **Generic-hyperparameter mapping**: each new proto field needs a string key
  registered in the GBT/RF hyperparameter spec (`GetGenericHyperParameterSpecification`)
  and a setter in the hyperparameter→proto mapping block. This is required for
  CLI, Python API, and hyperparameter tuner to expose them.

---

## Idea 1 (§5.1) — SNR-corrected split criterion for GBT

**Formula.** `gain_SNR = (Σg)² / (Σg² − (Σg)²/n_L + λ)` with rejection rule
`mean(g)² > var(g) / (n_L − 1)`. Requires a new per-bin statistic `Σg²`.

**Steps**

1. Extend bin statistic `FloatSumGradientHessian` (in
   `learner/decision_tree/splitter_accumulator.h` ~L100) with a new variant
   `FloatSumGradientHessianSquared<weighted>` carrying `sum_squared_gradient`.
   Keep the legacy struct so the path is unchanged when SNR is off.
2. Extend `LabelHessianNumericalScoreAccumulator` (~L749–L880) with optional
   `sum_squared_gradient` and a `ScoreSNR()` method computing the new gain plus
   a `PassesSnrThreshold()` predicate. Update `Add()`/`Sub()`/`Filler` paths.
3. Thread the extra statistic through:
   - histogram path: `FindSplitLabelRegressionFeatureNumericalHistogram` and
     the hessian regression entry point
     `FindBestConditionRegressionHessianGain` in `learner/decision_tree/training.cc`
     (~L640, L700, L2415).
   - exact path: `FindSplitLabelHessianRegressionFeatureNumericalCart` and
     categorical equivalent.
4. Add proto field on `DecisionTreeTrainingConfig`
   (`learner/decision_tree/decision_tree.proto`):
   `enum SplitScore { STANDARD = 0; SNR = 1; }` plus `optional SplitScore split_score = N;`.
   The accumulator dispatches `Score()` vs `ScoreSNR()` from this field.
5. Read the field in `FindBestConditionFromSplitterWorkRequest` (~L1222) and
   pass it down as part of the existing `dt_config` object (already
   propagated). Reject splits failing `PassesSnrThreshold()` by returning
   `kInvalidAttribute`.
6. Register a new generic hyperparameter `split_score` on both the GBT learner
   (`gradient_boosted_trees.cc` hyperparameter spec ~L2339) and the RF learner
   so it is reachable from CLI/Python.

**Tests**

1. Add `decision_tree_test.cc` cases mirroring `FindBestNumericalSplitCartBase`
   for the SNR path, including a "noisy leaf" fixture where the SNR criterion
   must reject a split that the standard one accepts.
2. Add a GBT end-to-end test in `gradient_boosted_trees_test.cc`: fit on a
   small regression dataset with `split_score=SNR`, assert convergence and
   smaller tree count than `STANDARD`.

---

## Idea 2 (§5.2) — SNR-corrected split criterion for Random Forest

**Formula.** `gain_SNR-RF = (Σy)² / (Σy² − (Σy)²/n_L)` with the same
rejection rule using label variance.

Crucially `LabelNumericalScoreAccumulator` already stores
`Σy, Σy², n` inside `utils::NormalDistributionDouble` — so **no new bin
statistic is needed** for RF regression.

**Steps**

1. Add `ScoreSNR()` and `PassesSnrThreshold()` to
   `LabelNumericalScoreAccumulator` (`splitter_accumulator.h` ~L611), reusing
   the existing distribution.
2. In `FindBestConditionRegression` (`training.cc` ~L825) dispatch on the
   `split_score` field added in Idea 1 to use either standard variance
   reduction or the SNR formula, and reject splits failing the threshold.
3. RF currently has no explicit split-criterion enum; reuse the
   `DecisionTreeTrainingConfig::SplitScore` enum from Idea 1 — same field
   serves both learners.
4. Classification labels are out of scope; document that `SplitScore::SNR`
   only affects numerical-label trees; with categorical labels it falls back
   to standard entropy and emits a one-time `LOG_WARNING`.

**Tests**

1. Decision-tree splitter unit test for the regression SNR path on a noise-
   only fixture (assert no split chosen).
2. End-to-end RF regression test in `random_forest_test.cc`: with SNR enabled,
   resulting trees have strictly fewer leaves than baseline on the same data.

---

## Idea 3 (§5.3) — Adaptive per-round learning rate for GBT

**Formulas.** Three variants:
- `EXACT`: `η_t = n / max_l v_{t,l}² · n_l` (needs leaf scan).
- `CONSERVATIVE`: `η_t = 1 / gain_t` (uses already-tracked total gain).
- `OPTIMISTIC`: `η_t = L_t / gain_t`.

All combined with an upper cap `eta_max` (defaulting to today's `shrinkage`
to behave as a clamp from above).

**Steps**

1. Add proto fields to `GradientBoostedTreesTrainingConfig`
   (`gradient_boosted_trees.proto`):
   `enum AdaptiveShrinkage { OFF=0; EXACT=1; CONSERVATIVE=2; OPTIMISTIC=3; }`
   `optional AdaptiveShrinkage adaptive_shrinkage = N [default = OFF];`
   `optional float adaptive_shrinkage_cap = N+1 [default = 0.1];`
2. After each tree is trained in the boosting loop
   (`gradient_boosted_trees.cc` ~L1002–L1014), before `UpdatePredictions`:
   - For **each** tree produced this round (multi-output: one η per tree),
     walk leaves reading `node.regressor().top_value()` and
     `node.num_pos_training_examples_without_weight()`.
   - Compute `max_l v² · n_l` (or use the cached gain for the conservative
     variant — the per-iteration sum-of-gains is already tracked for logging).
   - Compute `η_t` per tree, clamp to `[0, adaptive_shrinkage_cap]`.
3. Pass the per-iteration `η_t` into `UpdatePredictions` in place of the
   constant `gbt_config.shrinkage()`. Currently shrinkage is baked into the
   leaf value when `SetLeafValueWithNewtonRaphsonStep` runs
   (`loss_utils.cc` ~L55–L112) — refactor: leave the leaf value un-shrunk and
   apply the scalar inside `UpdatePredictions`, OR multiply the existing leaf
   by `η_t / gbt_config.shrinkage()` post-hoc (less invasive). Persist the
   adopted `η_t` in the model (new field on the GBT proto's per-iteration
   record) so inference reproduces it exactly.
4. Register hyperparameters `adaptive_shrinkage`, `adaptive_shrinkage_cap`.

**Tests**

1. Unit test: build a known tree, call the helper that computes `η_t`, assert
   numerical match against the closed form for all three variants.
2. End-to-end: train GBT with `adaptive_shrinkage=EXACT` and `OFF` on the same
   regression dataset; record `η_t` per iteration and assert monotone increase
   under EXACT (paper's prediction).
3. Inference parity: re-load the trained model, predict, and confirm match
   against in-memory predictions (validates the per-iteration `η_t`
   serialization).

---

## Idea 4 (§5.4) — Capacity constraint `T·(2^L − 1) ≤ n`

**Steps**

1. Add proto field to `DecisionTreeTrainingConfig` (or scoped on the GBT/RF
   proto if we want a per-learner switch):
   `enum CapacityPolicy { NONE=0; WARN=1; CLAMP_TREES=2; CLAMP_DEPTH=3; }`
   `optional CapacityPolicy capacity_policy = N [default = WARN];`
2. Compute the bound in the learner setup phase:
   `n_eff = n_train * subsample_or_bagging_fraction` (use
   `gbt_config.subsample()` for GBT, `bootstrap_size_ratio()` for RF).
   `max_T = floor(n_eff / (2^max_depth − 1))`.
   - For unlimited depth (`max_depth = -1`) emit a hard error if policy is
     `CLAMP_TREES` or `CLAMP_DEPTH` (bound is undefined).
3. Apply policy:
   - `WARN`: log a single `LOG_WARNING` if `T_user · (2^L − 1) > n_eff`.
   - `CLAMP_TREES`: set `num_trees = min(num_trees, max_T)` for GBT;
     for RF same (RF has no early-stopping default so the clamp is real).
   - `CLAMP_DEPTH`: set `max_depth = floor(log2(n_eff / num_trees + 1))`.
4. Surface as generic hyperparameter on both GBT and RF.

**Tests**

1. Unit test the helper `ComputeCapacityBound(n, L, b)` for a table of
   (n, L) pairs matching the table in §5.4.
2. End-to-end test: with `CLAMP_TREES`, request `num_trees=10000, max_depth=6`
   on a tiny dataset, assert the actual trained tree count equals `max_T`.
3. Test that `WARN` mode logs but does not change behavior.

---

## Idea 5 (§5.5) — Feature scoring via contrast dictionaries

Three scores, two pre-training, one post-training.

**Steps — pre-training scores (`s_f`, pairwise `G_fg^mean`)**

1. Add a new utility module
   `learner/decision_tree/contrast_dictionary.{h,cc}` that, given a
   `dataset::VerticalDataset` and a target column index, computes for each
   numerical/categorical feature the dictionary
   `{c_{f,k}}` from candidate split thresholds (reusing the existing
   `GenHistogramBins` to enumerate thresholds for consistency with the
   trainer).
2. Implement:
   - `IndividualSignal(feature_idx) → s_f` (max squared norm across
     thresholds).
   - `PairwiseRedundancyMean(f, g) → G_fg^mean`.
   - `SubspaceOverlap(f, g)` via SVD on `U_f^T U_g` (cap dictionary size with
     a sampled-thresholds option so the SVD stays affordable on
     high-cardinality features).
3. Expose these as a free-function CLI tool
   `cli/contrast_feature_scoring_main.cc` with output as a CSV/proto matrix.
4. Also wire into the existing `ComputeVariableImportancesFromAccumulatedPredictions`
   plumbing as new variable-importance kinds:
   - `INDIVIDUAL_SIGNAL_CONTRAST`
   - `PAIRWISE_REDUNDANCY_CONTRAST` (vector-valued, may need new schema)

**Steps — post-training score `A_f`**

1. Add an optional logger in the GBT loop
   (`gradient_boosted_trees.cc` ~L983 right after `UpdateGradients`) that
   reads the current normalized residual vector `r_t / ||r_t||` and, for each
   feature, accumulates `Σ_k (ĉ_{f,k}^T ĝ_t)²`.
2. Persist the running totals in the GBT model under a new repeated proto
   field `cumulative_residual_alignment` keyed by feature index.
3. Surface as variable importance kind `CUMULATIVE_RESIDUAL_ALIGNMENT`.
4. Gate the whole thing behind
   `gbt_config.compute_contrast_variable_importances` (default false) since
   it adds an `O(Σ_f N_f · n)` cost per round.

**Tests**

1. Unit test on a synthetic dataset with two perfectly redundant features:
   `G_fg^mean` ≈ 1, subspace overlap = 1.
2. Unit test on two orthogonal features: both ≈ 0.
3. End-to-end GBT test: train on noiseless tabular data, assert `A_f` ranking
   recovers the ground-truth feature ordering and is reproducible across
   seeds.

---

## Idea 6 (§5.6) — Matching pursuit (stumps) with SNR-based stopping

This is a packaging of Ideas 1 + 3 + a new stopping rule, not a new algorithm.
Equivalence to matching pursuit is automatic when `max_depth=1` and the SNR
criterion is on; the only addition is the stopping rule.

**Steps**

1. Extend the `EarlyStopping` enum in `gradient_boosted_trees.proto` with a
   new value `SIGNAL_CHANNEL_SATURATION`.
2. Extend the splitter's return path to surface "no candidate passed the SNR
   threshold" distinctly from "no improvement found" — already representable
   by `SplitSearchResult::kInvalidAttribute`, but the GBT loop currently
   silently keeps a stub root. Track "tree had zero accepted splits" as a
   boolean per iteration.
3. In the boosting loop, when policy is `SIGNAL_CHANNEL_SATURATION` and the
   newly trained tree(s) made no accepted split (a "trivial" tree), terminate
   training. Implement in `gradient_boosted_trees.cc` next to the existing
   `EarlyStopping` block.
4. Add an opt-in convenience preset `learner_kwargs={"matching_pursuit": True}`
   in the CLI/Python wrappers that sets `max_depth=1`, `split_score=SNR`,
   `adaptive_shrinkage=EXACT`, `early_stopping=SIGNAL_CHANNEL_SATURATION`,
   `subsample=1.0`.

**Tests**

1. Unit test the saturation predicate on a synthetic loop.
2. End-to-end stumps on a small regression dataset:
   - Train with `early_stopping=SIGNAL_CHANNEL_SATURATION`.
   - Compare the chosen `T*` against an explicit
     `MIN_VALIDATION_LOSS_ON_FULL_MODEL` cross-validation: paper predicts
     within 1 standard deviation.
3. Equivalence test: train MP-mode, also reconstruct predictions by running
   matching pursuit explicitly via NumPy on the binarized contrast matrix;
   require near-bitwise agreement on a tiny dataset.

---

## Phasing & dependencies

- **Phase A (foundation)** — parallel-safe:
  - Idea 1 (GBT SNR) — touches splitter accumulator + GBT histogram path.
  - Idea 2 (RF SNR) — depends on the proto enum from Idea 1, but otherwise
    independent of the GBT histogram changes.
  - Idea 4 (capacity) — purely config / setup-time logic, no splitter
    changes; can be done in parallel with 1 & 2.

- **Phase B (boosting-loop changes)** — depends on Phase A:
  - Idea 3 (adaptive `eta`) — needs a shrinkage refactor in the boosting
    loop; do before Idea 6.
  - Idea 5 pre-training pieces (`s_f`, `G_fg`) — fully independent, can land
    in parallel with Phase A or B.

- **Phase C (composition)** — depends on Ideas 1 & 3:
  - Idea 6 (MP + saturation stopping) — wires together earlier work and adds
    the new `EarlyStopping` value.
  - Idea 5 post-training piece `A_f` — depends on the residual-logging hook
    in Phase B (same callsite as adaptive-η), so do it alongside Idea 3.

## Verification (whole project)

1. `bazel test //yggdrasil_decision_forests/learner/decision_tree/...`
2. `bazel test //yggdrasil_decision_forests/learner/random_forest/...`
3. `bazel test //yggdrasil_decision_forests/learner/gradient_boosted_trees/...`
4. CLI smoke: `cli/train` with each new hyperparameter on a sample dataset.
5. Full-doc benchmark replication of E1–E6 from §6 of `ntk_doc.md`
   (deferred — companion paper).

## Out of scope

- Non-squared losses / multi-class (paper defers).
- Distributed training paths (focus on in-memory `TrainWithStatusImpl`; the
  distributed trainer would need a separate plumbing pass).
- GPU/SIMD optimization of the new histogram statistic.
