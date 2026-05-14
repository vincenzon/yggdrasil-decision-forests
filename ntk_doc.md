# The Neural Tangent Kernel in Random Forests and Gradient Boosted Models

**Abstract.** We translate the empirical-tangent-kernel framework of Litman & Guo (2026) from deep networks to tree ensembles. The cumulative dissipation operator $W_S$, the propagator $P_g$, the signal channel / reservoir decomposition, and the reservoir-invisibility identity $\ker W_S \subseteq \ker G_Q$ all admit clean closed-form expressions in random forests of stumps, full-depth random forests, and gradient boosted forests. The tree-ensemble setting is in fact simpler than the neural network setting: leaf indicators provide an orthogonal basis, the propagator factorizes as a product of rank-$L$ updates, and the spectral separation between signal and reservoir is exact rather than asymptotic. Using this structure we derive five practical modifications: an SNR-corrected split criterion for both GBM ($\sum g_i^2$) and RF ($\sum y_i^2$); an adaptive per-round learning rate $\eta_t = n / \max_l v_{t,l}^2 n_l$; a joint depth-and-iterations capacity constraint $T(2^L - 1) \leq n$; a contrast-dictionary feature-scoring framework; and a matching-pursuit interpretation of gradient boosted stumps with principled SNR-based stopping. Each modification corresponds to a different lever in the $W_S$ geometry. We outline an experimental program to test these on standard tabular benchmarks.

---

## 1. Introduction

The neural tangent kernel (Jacot et al., 2018) provides a kernel-machine interpretation of gradient descent in parameter space: for sufficiently wide networks, the linearization of the model around initialization is exact, and training reduces to kernel regression with a fixed kernel $K_\infty$. This frozen-kernel regime has been productive, but it is a sharp idealization. Real networks exhibit feature learning — the kernel evolves during training — and the resulting trajectories are not well-described by a single static kernel.

Litman & Guo (2026), in *A Theory of Generalization in Deep Learning*, propose an analysis that accommodates $O(1)$ kernel drift. Their central object is the cumulative dissipation operator

$$W_S = \int_0^T P_g(\tau, 0)^\top K_{SS}(\tau)\, P_g(\tau, 0)\, d\tau$$

which integrates the empirical tangent kernel along the actual training trajectory rather than at a single point. The range of $W_S$ defines a *signal channel* of directions the training process actively pushed, and its kernel defines a *reservoir* of directions training never touched. A reservoir-invisibility identity, $\ker W_S \subseteq \ker G_Q$, where $G_Q$ is the test-train transfer operator, then forces label noise that lives in the reservoir to be invisible to test predictions — providing a precise mechanism for benign overfitting (Bartlett et al., 2020; Belkin et al., 2019).

The framework is presented for neural networks but the underlying constructions are model-agnostic. The empirical tangent kernel $K_{SS}(t) = J_S(t) J_S(t)^\top$ requires only that the predictor be differentiable in parameters along the realized trajectory. Tree ensembles satisfy this: leaf values are smooth parameters and the leaf-membership matrix plays the role of the Jacobian.

This paper carries out that translation. We show that in tree ensembles the constructions of Litman & Guo simplify substantially. Leaf indicators are mutually orthogonal, the kernel is binary and sparse, the propagator factorizes as a product of low-rank stump updates, and the spectral separation between signal channel and reservoir is *exact*: eigenvalues of $K_{SS}^{\text{eff}}$ are either $n$ (signal) or $0$ (reservoir), with no intermediate values. This sharper structure makes the benign overfitting result cleaner and the algorithmic consequences more direct.

Our contributions:

1. We give closed-form expressions for $W_S$, $P_g$, and the signal channel in three tree-ensemble settings: stump forests, full random forests, and gradient boosted forests.
2. We identify the random forest as the frozen-kernel regime and gradient boosting as the feature-learning regime in the precise sense of Litman & Guo.
3. We derive an SNR-corrected split criterion — the paper's update rule applied at the leaf level — for both GBM and RF. The criterion requires one additional histogram statistic.
4. We derive an adaptive per-round learning rate $\eta_t = n / \max_l v_{t,l}^2 n_l$ from the propagator stability condition.
5. We derive a capacity constraint $T(2^L - 1) \leq n$ that bounds the signal channel dimension.
6. We introduce a feature-scoring framework based on the contrast dictionaries of each feature.
7. We show that gradient boosted stumps with the SNR criterion are exactly matching pursuit (Mallat & Zhang, 1993; Bühlmann & Yu, 2003) on the dictionary of binary contrast vectors, with a principled signal-channel-saturation stopping rule.

The proposed modifications are presented as theoretical predictions of the framework. Experimental validation on standard benchmarks is deferred to a companion paper.

## 2. Related Work

**Neural tangent kernel.** Jacot et al. (2018) introduced the NTK and established its convergence in the infinite-width limit. Lee et al. (2019) developed the wide-network Gaussian process correspondence. The NTK regime explains optimization but is too rigid for feature learning. Yang & Hu (2021) and Bordelon et al. (2024) study finite-width corrections; Litman & Guo (2026) reframe the question by integrating along the trajectory rather than fixing a kernel.

**Benign overfitting.** Bartlett et al. (2020) established benign overfitting in linear regression: training to interpolation does not harm test error when the data-covariance spectrum decays appropriately. Belkin et al. (2019) documented the double-descent curve; Hastie et al. (2022) extended these results. Litman & Guo unify these phenomena as different spectral filters on a single Gramian (their Theorem H.1).

**Kernel views of tree ensembles.** Breiman (2000) noted that random forests define a kernel through co-leaf membership. Scornet (2016) made this precise and proved consistency under regularity conditions. Davies & Ghahramani (2014) studied random forest kernels for Bayesian inference. None of these works analyze the kernel as evolving during training, which is the central object of $W_S$.

**Boosting as matching pursuit.** Bühlmann & Yu (2003) showed that L2Boosting with stumps is equivalent to matching pursuit (Mallat & Zhang, 1993; Pati et al., 1993) on the dictionary of stump indicators. Bühlmann (2006) extended this to other base learners. Our matching-pursuit-with-SNR-stopping result extends this connection by giving boosting a principled, cross-validation-free stopping rule rooted in the spectral structure of $W_S$.

**Significance-based splitting.** Hothorn et al. (2006) introduced conditional inference trees, which decide splits via permutation tests rather than gain maximization. This is the closest existing analog to our SNR-corrected criterion. The difference: their threshold is a fixed significance level (a hyperparameter); ours is derived from the propagator-stability condition and is parameter-free. Wager & Athey (2018) impose a similar "honesty" criterion in causal forests, requiring that treatment-effect estimates in leaves are statistically meaningful before splitting.

**Gradient boosting.** Friedman (2001) introduced gradient boosting as functional gradient descent. Chen & Guestrin (2016) and Ke et al. (2017) developed the histogram-based implementations (XGBoost, LightGBM) that we target. Our proposed modifications integrate into the LightGBM histogram algorithm with one additional sufficient statistic per bin.

**Generalization of tree ensembles.** Mentch & Hooker (2016) developed subsampling-based confidence intervals for random forests. Athey et al. (2019) studied asymptotic properties of generalized random forests. These works focus on statistical inference rather than the kernel-geometric structure we examine.

## 3. Background: The Litman & Guo Framework

We summarize the constructions used in the rest of the paper. Let $u_S(t) \in \mathbb{R}^n$ be the vector of training predictions at training time $t$, with $u_S(0)$ the initialization. Define the parameter Jacobian $J_S(t) = \partial u_S / \partial w \big|_{w(t)}$, the empirical tangent kernel $K_{SS}(t) = J_S(t) J_S(t)^\top$, and the loss Hessian in output space $B(t) = \partial_u^2 \Phi_S$.

The gradient $g(t) = \nabla_u \Phi_S(u(t))$ evolves as $\partial_t g = -B(t) K_{SS}(t) g$. The *propagator* $P_g(t, s)$ is the fundamental matrix solution:

$$\partial_t P_g(t, s) = -B(t) K_{SS}(t) P_g(t, s), \qquad P_g(s, s) = I$$

The *cumulative dissipation* operator is

$$W_S = \int_0^T P_g(\tau, 0)^\top K_{SS}(\tau)\, P_g(\tau, 0)\, d\tau \tag{1}$$

The *signal channel* is $\text{range}(W_S)$; the *reservoir* is $\ker W_S$. The test-train transfer operator $G_Q = \int_0^T K_{QS}(\tau) P_g(\tau, 0)\, d\tau$ inherits the factor $J_S^\top$ through $K_{QS} = J_Q J_S^\top$, which forces the reservoir-invisibility identity

$$\ker W_S \subseteq \ker G_Q \tag{2}$$

(Litman & Guo, 2026, Proposition 3.2).

The factorization $D = C_S W^{1/2}$, $G = C_Q W^{1/2}$ (their Theorem E.2) leads to the transfer-error decomposition (Theorem E.3): every linear predictor $A$ satisfies

$$(G - AD) W^\dagger (G - AD)^\top = R_\perp R_\perp^\top + (A - A_\circ) D W^\dagger D^\top (A - A_\circ)^\top$$

with optimal predictor $A_\circ = G W^\dagger D^\top (D W^\dagger D^\top)^\dagger$. Under isotropic noise the interpolation variance reduces to

$$R_\rho = \sigma^2 \sum_{j} \frac{\|J_Q(w_0) v_j\|^2}{\sigma_j^2} \tag{3}$$

(Corollary H.4). Benign overfitting holds when noise lies in $\ker W_S$, so that $0/0$ terms in (3) drop out by (2) and the remaining sum is bounded.

## 4. The NTK Structure in Tree Ensembles

### 4.1 The Empirical Tangent Kernel in Trees

For a tree ensemble, the differentiable parameters at round $t$ are the leaf values $\{v_{t,l}\}_{l=1}^{L_t}$. The prediction for example $i$ is $F_t(x_i) = \sum_{s \leq t} \eta v_{s, \ell_s(x_i)}$ where $\ell_s(x_i)$ is the leaf example $i$ occupies in tree $s$. The Jacobian of $u_S(t)$ with respect to round-$t$ leaf values is a binary indicator matrix; the kernel at round $t$ is

$$[K_{SS}^{(t)}]_{ij} = \sum_l v_{t,l}^2 \cdot \mathbf{1}[\ell_t(x_i) = l] \cdot \mathbf{1}[\ell_t(x_j) = l] \tag{4}$$

Two examples are coupled at round $t$ iff they occupy the same leaf, weighted by the squared leaf value. The kernel is rank-$L_t$ (at most), with eigenvectors equal to the leaf indicator vectors $\mathbf{1}_l$ and eigenvalues $v_{t,l}^2 n_l$.

**Two-level coupling.** The loss is example-separable, so $B = \text{diag}(h_1, \ldots, h_n)$ is diagonal. However, the *split selection* couples examples through the gain criterion:

$$\text{gain} = \frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda} - \frac{(\sum_i g_i)^2}{\sum_i h_i + \lambda}$$

The squared sums $(\sum_{i \in L} g_i)^2 = \sum_{i, j \in L} g_i g_j$ contain explicit cross terms between every pair of examples in the candidate child. This is the off-diagonal coupling that distinguishes feature learning (kernel is shaped by data) from frozen-kernel regimes (kernel is fixed by initialization). The split decision selects the partition that maximizes within-group gradient alignment — the kernel is actively shaped to point at directions of highest coherent signal.

### 4.2 Random Forest of Stumps

Each tree is a single split. The leaf value is the mean label in the leaf: $v_{t,l} = \bar{y}_l$, independent of the current model output $u_S(t)$. The kernel at round $t$ contributes a rank-2 update:

$$K_{SS}^{(t)} = \bar{y}_L^2 \mathbf{1}_L \mathbf{1}_L^\top + \bar{y}_R^2 \mathbf{1}_R \mathbf{1}_R^\top$$

where $\mathbf{1}_L, \mathbf{1}_R$ are indicator vectors of the left and right children. Since each tree's kernel depends only on $(x_i, y_i)$ and not on $u_S(t)$, this is the *frozen-kernel regime* in the precise sense of Litman & Guo: the kernel sequence $\{K_{SS}^{(t)}\}$ can be computed before any training round is executed.

The propagator is a product of orthogonal rank-2 updates:

$$P_g(T, 0) = \prod_{t=0}^{T-1} \left( I - \eta B \cdot K_{SS}^{(t)} \right)$$

Because all leaf indicator vectors are mutually orthogonal *within a tree*, and stumps have only two leaves, each tree's update factors cleanly through 2D leaf subspaces.

The signal channel is the span of all $\mathbf{1}_{L_t}, \mathbf{1}_{R_t}$ across $t = 0, \ldots, T-1$. With $m$ features and $N$ unique values per feature, there are at most $m(N-1)$ distinct stumps, each contributing one contrast direction. The dictionary is overcomplete (more directions than dimensions of $\mathbb{R}^n$) when $m(N - 1) > n - 1$.

### 4.3 Full Random Forest

The kernel structure generalizes. Each tree of depth $L_t$ has $2^{L_t}$ leaves (or up to that many in unbalanced trees). The kernel contribution is rank-$L_t$ at most:

$$K_{SS}^{(t)} = \sum_{l=1}^{L_t} v_{t,l}^2 \mathbf{1}_l \mathbf{1}_l^\top$$

with $v_{t,l} = \bar{y}_l$ and the leaf indicators $\mathbf{1}_l$ mutually orthogonal *within tree $t$* (each example occupies exactly one leaf). Across trees the indicator vectors are not orthogonal — that is what makes the signal channel grow.

**Path-dependence: subtle but present.** Although a random forest's tree *structures* depend on labels only through the bootstrap sample, the leaf *values* depend on labels through $\bar{y}_l$. The kernel $K_{SS}^{(t)}$ at tree $t$ is independent of $u_S(t)$ but is fully determined before round $t$ — distinguishing this from the gradient boosting case where $K_{SS}^{(t)}$ depends on $u_S(t-1)$ through the residuals. So the random forest is in the *frozen-kernel regime* with respect to the trajectory, even though its kernel depends on labels.

**Signal/reservoir structure.** A direction $v \in \mathbb{R}^n$ is in the reservoir iff

$$v^\top W_S v = \eta^2 \sum_{t, l} v_{t,l}^2 \left( \sum_{i \in l} v_i \right)^2 = 0$$

meaning no tree produced leaf patterns that resolve $v$, weighted by leaf-value magnitude. Identical-feature examples have $e_i - e_j$ permanently in the reservoir; patterns requiring more depth than available are in the reservoir; patterns affecting fewer than `min_data_in_leaf` examples are in the reservoir.

### 4.4 Gradient Boosted Forests

This is the feature-learning regime. Tree $t$ is fit to the residuals $r_t = y - u_S(t)$, so the splits and leaf values both depend on $u_S(t-1)$. Leaf values for squared loss are $v_{t,l} = \bar{r}_l^{(t)}$; for general loss they are $v_{t,l} = -\bar{g}_l / \bar{h}_l$ where $g_i, h_i$ are gradient and Hessian.

The kernel is structurally the same as (4), but the leaf indicators $\mathbf{1}_l$ at round $t$ depend on the trajectory $u_S(0), \ldots, u_S(t-1)$ through the residuals that drove split selection. The signal channel built up by $W_S$ is path-dependent: different initializations or sample orderings give different signal channels.

**Discrete propagator.** With learning rate $\eta$ and squared loss,

$$P_g(T, 0) = \prod_{t=0}^{T-1} \left( I - \frac{\eta}{n} \sum_l v_{t,l}^2 \mathbf{1}_l^{(t)} \mathbf{1}_l^{(t)\top} \right) \tag{5}$$

Within a single tree the leaf indicators are mutually orthogonal, so each tree's factor acts diagonally on its own leaf subspaces with eigenvalues $\eta v_{t,l}^2 n_l / n$ on $\text{span}(\mathbf{1}_l)$ and 1 elsewhere. Stability requires each eigenvalue in $[0, 1]$, giving the per-round constraint $\eta \leq n / \max_l v_{t,l}^2 n_l$.

**Spectral separation is sharp.** On the signal channel, repeated application of the factors drives eigenvalues of $P_g$ toward 0; on the reservoir, $P_g = I$ exactly because $K_{SS}^{(t)} \mathbf{1}_{\text{reservoir}} = 0$ for every $t$. So eigenvalues of $K_{SS}^{\text{eff}} = n(I - P_g(T, 0))$ are exactly $0$ on the reservoir and approach $n$ on the signal channel. The intermediate eigenvalues that complicate the deep-network analysis are absent here.

### 4.5 The Frozen vs. Feature-Learning Dichotomy

| Property | Random forest | Gradient boosting |
|---|---|---|
| Tree structure depends on $u_S(t)$ | No | Yes (through residuals) |
| Leaf values depend on $u_S(t)$ | No | Yes |
| Kernel sequence pre-computable | Yes | No (path-dependent) |
| Regime (Litman & Guo) | Frozen | Feature learning ($O(1)$ drift) |
| Signal channel determinism | Fixed by data + structures | Trajectory-dependent |

This puts the random forest exactly in the original NTK regime (Jacot et al., 2018), while the gradient boosted forest exemplifies the $O(1)$-drift regime that Litman & Guo's $W_S$ framework was developed to handle.

## 5. Practical Improvements Motivated by $W_S$ Geometry

### 5.1 SNR-Corrected Split Criterion (GBM)

The standard LightGBM split criterion is $\text{gain} = (\sum_{i \in L} g_i)^2 / (\sum_{i \in L} h_i + \lambda) + \ldots$ The denominator measures loss curvature in the leaf — how steep the loss surface is — not whether the per-example gradients are coherent or scattered. Litman & Guo's practical algorithm updates parameter $k$ only when $\mu_k^2 > \sigma_k^2 / (b - 1)$ — only when the mean gradient exceeds the noise level (their Section 4). Applied at the leaf level:

$$\text{gain}_{\text{SNR}} = \frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} g_i^2 - (\sum_{i \in L} g_i)^2 / n_L + \lambda} + (\text{right child term}) \tag{6}$$

The denominator is $(n_L - 1) \hat{\sigma}_L^2$ — the within-leaf gradient variance. Split iff $\bar{g}_L^2 > \hat{\sigma}_L^2 / (n_L - 1)$.

**Effect.** Early in training, residuals are large and coherent; both criteria agree. Late in training, residuals are small and dominated by idiosyncratic noise; the SNR criterion refuses splits the standard criterion would make. This is *data-driven early stopping at the split level*, suppressing splits that would expand the signal channel into noise directions where (3) becomes large.

**Implementation cost.** One additional histogram per feature: $\sum g_i^2$ per bin. Within-leaf variance is then recoverable from the three sufficient statistics $(\sum g, \sum h, \sum g^2)$ per leaf. This is "one extra state vector" — the cost Litman & Guo identify for their practical algorithm.

### 5.2 SNR-Corrected Split Criterion (Random Forest)

For random forests, the leaf value is $\bar{y}_l$ and enters $W_S$ as $v_{t,l}^2 = \bar{y}_l^2$. The same SNR question — is the mean signal large relative to within-leaf noise — applies, but now to labels rather than gradients:

$$\text{gain}_{\text{SNR-RF}} = \frac{(\sum_{i \in L} y_i)^2}{\sum_{i \in L} y_i^2 - (\sum_{i \in L} y_i)^2/n_L} + (\text{right child term}) \tag{7}$$

with the additional statistic $\sum y_i^2$ per histogram bin. This is closely related to but distinct from the conditional-inference-tree criterion of Hothorn et al. (2006): they use permutation tests with a fixed significance threshold, while (7) is derived from the propagator-stability condition.

### 5.3 Adaptive Per-Round Learning Rate

From (5), the propagator factor at round $t$ has eigenvalues $\eta v_{t,l}^2 n_l / n$ on leaf subspaces and 1 elsewhere. Stability requires these in $[0, 1]$:

$$\eta_t = \frac{n}{\max_l v_{t,l}^2 n_l} \tag{8}$$

This is the tightest learning rate with guaranteed propagator stability. Both $v_{t,l}^2$ and $n_l$ are available immediately after tree construction; the cost is one pass over the $L_t$ leaves.

**Conservative simplification.** Using $\max_l v_{t,l}^2 n_l \leq n \cdot \text{gain}_t$ (max $\leq$ sum) gives the conservative bound $\eta_t \geq 1 / \text{gain}_t$, which requires no leaf-level statistics — only the total gain already computed during tree construction. Using $\max \geq$ average gives the optimistic bound $\eta_t \leq L_t / \text{gain}_t$, which is tight for balanced trees.

**Behavior over training.** Since gain decreases as residuals shrink, $\eta_t$ *increases* over rounds — the opposite of decaying-schedule heuristics. Larger steps are admissible late in training when only weak corrections remain. The framework also justifies clamping: in practice combine (8) with an upper cap $\eta_{\max}$ to prevent over-stepping in early rounds when leaf values can be artificially large.

### 5.4 Capacity Constraint $T(2^L - 1) \leq n$

The signal channel dimension is bounded by the combinatorics of the leaf decomposition. For $T$ trees of depth $L$, each contributes at most $2^L - 1$ linearly independent directions (since the $2^L$ leaf indicators sum to $\mathbf{1}$). The total signal channel dimension is bounded by

$$\dim(\text{signal channel}) \leq \min(n - 1, T(2^L - 1)) \tag{9}$$

When $T(2^L - 1) \leq n$, the signal channel cannot span $\mathbb{R}^n$ — a non-trivial reservoir is guaranteed independent of the data. This gives a principled joint constraint on $T$ and $L$:

| Depth $L$ | Max trees before saturation |
|---|---|
| 1 (stumps) | $n$ |
| 2 | $n/3$ |
| 3 | $n/7$ |
| $L$ | $n/(2^L - 1)$ |

Rather than searching $T$ and $L$ independently, search points on the boundary $T(2^L - 1) = n$ or strictly below it. Points above this curve risk saturating the signal channel and fitting noise. With `bagging_fraction` $= b$, the effective constraint tightens to $T(2^L - 1) \leq bn$.

### 5.5 Feature Scoring via Contrast Dictionaries

For each feature $f$ with $N_f$ unique values, define the *contrast dictionary*

$$c_{f,k} = \bar{y}_{L_{f,k}} \mathbf{1}_{L_{f,k}} - \bar{y}_{R_{f,k}} \mathbf{1}_{R_{f,k}}, \quad k = 1, \ldots, N_f - 1$$

one vector per candidate split threshold. Let $V_f = \text{span}\{c_{f,k}\}_k$ and $\hat{c}_{f,k} = c_{f,k} / \|c_{f,k}\|$. Three scores decompose feature importance into orthogonal questions:

**Individual signal:** $s_f = \max_k \|c_{f,k}\|^2$ — max gain achievable by feature $f$ against labels. Pre-training.

**Pairwise redundancy (mean):** $G_{fg}^{\text{mean}} = \frac{1}{(N_f - 1)(N_g - 1)} \sum_{k, l} (\hat{c}_{f,k}^\top \hat{c}_{g,l})^2$. Average squared cosine across all split pairs — robust to single accidental collinearities. Pre-training.

**Subspace overlap:** $\|P_{V_f} P_{V_g}\|_F^2 = \sum_i \sigma_i^2$ where $\sigma_i$ are singular values of $U_f^\top U_g$ (orthonormal bases). The largest $\sigma_i^2$ is the squared cosine of the smallest principal angle and equals $\max_{k,l} (\hat{c}_{f,k}^\top \hat{c}_{g,l})^2$. Two features are non-redundant iff $V_f \perp V_g$.

**Cumulative residual alignment:** $A_f = \sum_{t=0}^{T-1} \frac{1}{N_f - 1} \sum_k (\hat{c}_{f,k}^\top \hat{r}_t)^2$ — total mean alignment of feature $f$'s split dictionary with residuals across training. A post-training importance measure that integrates evidence from *all* candidate splits of every feature at every round, in contrast to total-gain importance which records only the winning split per round.

Standard total-gain importance conflates $s_f$, $G_{fg}$, and $A_f$. A feature with high gain but high overlap with another feature is not adding as much new signal channel capacity as its gain suggests.

### 5.6 Matching Pursuit with SNR-Based Stopping

Gradient boosted stumps with the SNR-corrected criterion (Section 5.1) admit a clean reinterpretation. The set of $m(N - 1)$ contrast vectors from all features and thresholds forms an overcomplete dictionary in $\mathbb{R}^n$. Each boosting round selects the dictionary element most aligned with the current residual — exactly matching pursuit (Mallat & Zhang, 1993). The SNR threshold provides a stopping criterion: terminate when no remaining contrast direction has $\text{SNR} > 1$.

Bühlmann & Yu (2003) established the L2Boosting / matching pursuit equivalence without an intrinsic stopping rule (their stopping is by cross-validation). The $W_S$ framework supplies one: *the signal channel saturates when no remaining direction has signal distinguishable from noise*. This is the same criterion as (6) applied uniformly — stop when every candidate split would be refused.

**Connection to OMP.** Orthogonal matching pursuit (Pati et al., 1993) re-fits all selected dictionary coefficients after each addition. Standard boosting does not re-fit; its analog would be cyclic boosting or fully-corrective boosting (Mukherjee et al., 2013). The choice affects how greedily the signal channel is expanded.

## 6. Experimental Plan

We outline experiments to test the proposed modifications. Implementation will target LightGBM (Ke et al., 2017) with one additional histogram statistic per feature.

**E1. SNR-corrected split criterion (GBM).** Compare LightGBM's standard criterion against (6) on standard tabular benchmarks (Higgs, MSLR, KKBox, Yahoo LTR). Measure test error, training rounds to convergence, and signal channel dimension (rank of $W_S$ to numerical tolerance). Hypothesis: (6) reduces overfitting at large $T$ without harming bias-limited regimes.

**E2. SNR-corrected criterion (RF).** Compare scikit-learn's standard criterion against (7) on the same benchmarks. Hypothesis: (7) gives smaller trees with comparable accuracy by refusing splits where leaf-value estimates are noise.

**E3. Adaptive learning rate.** Replace fixed $\eta$ with (8) (and the conservative $1/\text{gain}_t$ variant). Hypothesis: matches or exceeds tuned-$\eta$ baselines, with the conservative variant within $\epsilon$ of the exact form. Plot $\eta_t$ over rounds — predict monotone increase.

**E4. Capacity constraint.** Grid-search $(T, L)$ above and below the curve $T(2^L - 1) = n$. Hypothesis: points above the curve show classical overfitting (test error rises beyond optimal $T$); points below do not, regardless of $T$.

**E5. Feature scoring.** Compare $s_f$, $G_{fg}^{\text{mean}}$, and $A_f$ against LightGBM's total-gain importance for feature selection. Hypothesis: redundancy-adjusted $A_f - \alpha \sum_g G_{fg}^{\text{mean}} A_g$ selects fewer features with comparable accuracy.

**E6. Matching pursuit stopping.** Boost stumps with (6) and no cross-validation. Hypothesis: the SNR-stop $T$ matches the cross-validated optimum within one standard deviation, eliminating the need for held-out validation.

For each experiment we will report 5-seed means and standard errors, total training time, and signal channel dimension where applicable.

## 7. Discussion

The Litman & Guo framework was developed to handle the kernel drift characteristic of feature learning in deep networks. Tree ensembles exhibit a sharper, more analyzable form of the same structure: orthogonal leaf indicators give exact spectral separation, the propagator factorizes as a product of low-rank updates, and the reservoir is exactly $\bigcap_t \ker K_{SS}^{(t)}$. The framework's predictions translate cleanly: benign overfitting in trees occurs when label noise lands in directions no tree can resolve, controlled by `min_data_in_leaf`, depth, and the SNR criterion.

The practical modifications we derived — SNR-corrected splits, adaptive learning rate, capacity constraint, contrast-dictionary scoring, matching-pursuit stopping — each correspond to a single lever in the geometry. SNR criteria control which directions enter the signal channel; adaptive learning rate controls how fast each direction is fit; capacity constraints bound the signal channel dimension; contrast scoring decomposes feature importance into the geometric primitives of $W_S$; matching pursuit reframes boosting as greedy signal channel expansion with a principled stopping rule.

The most striking structural difference from the deep network case is the sharpness of the signal/reservoir partition. In the wide-network NTK limit, eigenvalues of the trajectory-integrated kernel span a continuum; benign overfitting requires careful balance between signal and noise eigenspaces. In tree ensembles the partition is binary: eigenvalues of $K_{SS}^{\text{eff}}$ are exactly $n$ or exactly $0$. This makes the reservoir-invisibility identity (2) hold with equality on the reservoir, and (3) reduces to a finite sum over signal-channel modes. Tree ensembles, in this sense, are the cleanest realization of the Litman & Guo framework.

**Limitations.** All derivations assume squared loss for simplicity. Extensions to log-loss and other losses require tracking the Hessian $B(t) = \text{diag}(p_i(1-p_i))$, which evolves during training. The propagator factorization (5) is loss-agnostic but the eigenvalue analysis for the adaptive learning rate requires explicit $B$. Multi-class classification introduces additional structure we do not analyze. The capacity bound (9) is a worst-case combinatorial bound; in practice the SNR criteria leave many candidate directions unused, making the effective signal channel dimension smaller than $T(2^L - 1)$.

## References

Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2), 1148–1178.

Bartlett, P. L., Long, P. M., Lugosi, G., & Tsigler, A. (2020). Benign overfitting in linear regression. *PNAS*, 117(48), 30063–30070.

Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. *PNAS*, 116(32), 15849–15854.

Bordelon, B., Atanasov, A., & Pehlevan, C. (2024). A dynamical model of neural scaling laws. *ICML*.

Breiman, L. (2000). Some infinity theory for predictor ensembles. *Technical Report 577*, UC Berkeley.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Bühlmann, P. (2006). Boosting for high-dimensional linear models. *Annals of Statistics*, 34(2), 559–583.

Bühlmann, P., & Yu, B. (2003). Boosting with the L2 loss: regression and classification. *Journal of the American Statistical Association*, 98(462), 324–339.

Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD*.

Davies, A., & Ghahramani, Z. (2014). The random forest kernel and other kernels for big data from random partitions. *arXiv:1402.4293*.

Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139.

Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

Hastie, T., Montanari, A., Rosset, S., & Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. *Annals of Statistics*, 50(2), 949–986.

Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased recursive partitioning: a conditional inference framework. *Journal of Computational and Graphical Statistics*, 15(3), 651–674.

Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: convergence and generalization in neural networks. *NeurIPS*.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*.

Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Sohl-Dickstein, J., & Pennington, J. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. *NeurIPS*.

Litman, R., & Guo, B. (2026). A theory of generalization in deep learning. *arXiv:2605.01172*.

Mallat, S., & Zhang, Z. (1993). Matching pursuit with time-frequency dictionaries. *IEEE Transactions on Signal Processing*, 41(12), 3397–3415.

Mentch, L., & Hooker, G. (2016). Quantifying uncertainty in random forests via confidence intervals and hypothesis tests. *JMLR*, 17(26), 1–41.

Mukherjee, I., Rudin, C., & Schapire, R. E. (2013). The rate of convergence of AdaBoost. *JMLR*, 14(1), 2315–2347.

Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). Orthogonal matching pursuit: recursive function approximation with applications to wavelet decomposition. *Asilomar Conference*.

Scornet, E. (2016). Random forests and kernel methods. *IEEE Transactions on Information Theory*, 62(3), 1485–1500.

Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*, 113(523), 1228–1242.

Yang, G., & Hu, E. J. (2021). Feature learning in infinite-width neural networks. *ICML*.
