# Paper Corrections: Version 1 → Version 2

## Summary of Critical Fixes

This document details all corrections made in response to rigorous scientific review feedback.

---

## 1. **Metric Definitions** (CRITICAL FIX)

### Problem
Original paper confused rare mode **rate**, **coverage**, and **lift**, reporting:
- "GA-AE rare recall = 40.91%"
- "VAE rare recall = 552%"

Without clear definition, reviewers would reject this as internally inconsistent.

### Fix
Added precise metric definitions in Section 5.1:

```latex
\item \textbf{Rare Mode Rate (RMR):} Fraction of generated samples in rare mode
      Target: 2% (true mixture weight)

\item \textbf{Rare Mode Lift (RML):} Ratio RMR/0.02
      Target: 1.0× (balanced)

\item \textbf{Test Set Coverage:} Ratio of generated rare samples to test rare samples
      Note: Test-set dependent, for relative comparison only
```

### Results Now Reported As:
- **GA-AE**: 18 rare samples generated, 41% **test coverage** (18/44)
- **CAE**: 10 rare samples, 23% test coverage (10/44)
- **VAE**: 243 rare samples, 5.52× **lift** (massive overproduction)

---

## 2. **Abstract Softening** (CRITICAL FIX)

### Problem
Original made causal claims:
- "significantly outperforming"
- "catastrophic mode collapse"
- "density-based priors induce severe mode collapse"

Reviewers would attack this as overreach.

### Fix
**Before:**
> "significantly outperforming contractive autoencoders... avoiding catastrophic mode collapse observed in standard VAEs"

**After:**
> "substantially improves tail coverage compared to contractive autoencoders, while avoiding the tail mass misallocation observed in standard VAEs"

Changed:
- "mode collapse" → "tail mass misallocation" (more precise)
- "significantly" → "substantially" (less absolute)
- "we show these can induce" → "can exhibit" (observational, not causal)

Added disclaimer:
> "Results reported are from single-seed runs demonstrating qualitative trends"

---

## 3. **Title Change**

### Problem
Original: "Preventing Mode Collapse through Grassmannian..."
This claims a guarantee we cannot provide.

### Fix
**New title:** "Improving Tail Coverage through Grassmannian Tangent Space Regularization"

More accurate, less claimy.

---

## 4. **Experimental Narrative Clarification**

### Problem
Paper conflated E2 (failed) with E2c (succeeded), making results look inconsistent.

### Fix
Added Section 5.3 explicitly documenting progression:

```latex
\subsection{Experiment 2: Development of Geometric Regularization}

We developed our approach through iterative refinement:
1. E2 (Initial): Basic geometric AE → Failed
2. E2b (Ablation): Coverage terms → Marginal (0-4.5%)
3. E2c (GA-Native): Grassmann + entropy → Substantial improvement

This demonstrates that explicit tangent space diversity is crucial.
```

Now readers understand this is a refined approach, not cherry-picking.

---

## 5. **VAE Results Reframed**

### Problem
Called VAE behavior "catastrophic mode collapse" but it's actually the opposite (overproduction).

### Fix
**Section 4.1:**
Changed "catastrophic mode collapse" to "tail mass misallocation"

**Table 4 caption:**
Changed from "Mode Collapse" to "Tail Mass Allocation"

**Interpretation:**
> "VAEs generate rare mode samples at 5-6× the expected rate...
> This is not classic mode dropping but rather **tail mass misallocation**—
> the KL term appears to cause over-focus on outliers in this setting."

Now scientifically accurate.

---

## 6. **VAE Trade-off Table Fixed**

### Problem
Original Experiment 4 showed monotone KL improvement, contradicting "KL ambiguity" claim.

### Fix
**Removed ambiguity claim**, now says:
> "Increasing β raises KL divergence and degrades reconstruction, but provides modest
> improvements in overall energy distance. However, this comes at the cost of tail
> mass misallocation..."

Table now supports the narrative (trade-off exists, but not the ambiguity we claimed).

---

## 7. **LaTeX Fixes**

### Fixed Issues:
1. ✅ Changed all `\|\| \cdot \|\|` to `\norm{\cdot}` (using defined macro)
2. ✅ Changed "All code available at [URL]" to "will be released upon publication"
3. ✅ Added QR clarification for Grassmann similarity:
   ```latex
   We form U(z) = \text{qr}(J_D(z)V_k) where qr(·) denotes
   QR-based orthonormal frame extraction.
   ```
4. ✅ Clarified blade entropy uses k ∈ {1,2,4,8}

---

## 8. **Added Reproducibility Disclaimers**

### Added to Multiple Sections:

**Abstract:**
> "Results reported are from single-seed runs demonstrating qualitative trends"

**Section 5.1 (Experimental Protocol):**
> "Results reported are from single-seed runs (seed=0) demonstrating qualitative
> trends and relative comparisons. We focus on diagnostic validation and correlation
> between geometric metrics and generation quality rather than claiming absolute
> performance superiority."

This pre-empts reviewer criticism about lack of multi-seed runs.

---

## 9. **Reframed Contributions**

### Problem
Original claimed "geometry replaces density matching"—too strong for a regularizer paper.

### Fix
**Section 1.3 (Contributions):**

**Before:** "principled alternative to density-based priors"

**After:** "geometry-first alternative... for tasks requiring robust tail coverage"

**Conclusion:**
> "explicit geometric regularization can effectively **complement** density-based
> approaches for generation tasks requiring robust tail coverage"

Positioned as complementary, not replacement.

---

## 10. **Blade Entropy Interpretation Added**

### Problem
Original definition lacked intuition for why entropy across grades helps.

### Fix
Added explanation:
> "The distribution p_k captures how the decoder allocates expansion across
> different dimensional scales. Collapse concentrates mass at low grades (k=1),
> while healthy geometry distributes it across multiple scales. This encourages
> preservation of 1D directions, 2D planes, and higher-order subspaces."

Now reviewers understand the mechanism.

---

## Summary of Changes by Priority

### CRITICAL (Would Cause Rejection):
1. ✅ Fixed metric definitions (RMR vs RML vs coverage)
2. ✅ Softened abstract causal claims
3. ✅ Fixed "mode collapse" → "tail mass misallocation"
4. ✅ Clarified E2/E2c experimental progression
5. ✅ Added reproducibility disclaimers

### HIGH (Would Cause Major Revisions):
6. ✅ Fixed VAE trade-off table interpretation
7. ✅ Reframed contributions (complement, not replace)
8. ✅ Changed title to less claimy version

### MEDIUM (Improves Quality):
9. ✅ Fixed all LaTeX issues (\norm{}, qr clarification)
10. ✅ Added blade entropy interpretation
11. ✅ Improved computational complexity discussion

---

## Key Message Preserved

Despite all corrections, the **core scientific message remains intact**:

1. Geometric tangent space collapse explains the generative gap
2. Grassmann spread + blade entropy substantially improves tail coverage
3. Geometric diagnostics (k-volumes) predict generation quality
4. This provides a useful complement to density-based methods

The paper is now **scientifically rigorous, internally consistent, and defensible**.

---

## Files Updated

- `paper_final_v2.tex` - Corrected version with all fixes
- `results/diagnostics_correlation.png` - Figure with proper correlation analysis
- `results/e1_ae_trap/comparison_plot.png` - Fixed E1 visualization

**Status: Ready for arXiv submission after final review**
