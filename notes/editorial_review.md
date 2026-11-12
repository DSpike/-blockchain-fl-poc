# Editorial review: FairPoC (ICBC 2027 submission)

Reviewer perspective: senior IEEE reviewer in blockchain systems and
federated learning. Reviewed 2026-11-12 against the full text at commit
b364b4e plus the title change.

**Recommendation as it stands: weak accept / borderline.** One genuinely
novel result, carefully bounded. Four issues below would be raised by any
careful reviewer and two of them are self-inflicted.

---

## MAJOR

### M1. The baseline set is inconsistent across the paper

Section VI-A names four baselines: FedAVG, PoCL, BLADE-FL, TMC-Shapley.
Tables II and III show three: FedAVG, PoCL, **FLTrust**.

- **FLTrust is never introduced and never cited.** It appears cold in
  Table II, is discussed once ("FLTrust sacrifices fairness for Byzantine
  robustness"), and has no entry in the baseline list or bibliography.
- **TMC-Shapley is described in the baseline list and evaluated nowhere.**
  It appears exactly once in the paper, at line 463.
- **BLADE-FL is promised in the baseline list but absent from the main
  results.** It appears only in the adversarial table.

This is the first thing a reviewer checks and it fails on all three
counts. It reads as results assembled from more than one experimental
campaign without reconciling the text. Fix before submission; it is
cheap and it is damaging.

### M2. A claim stated twice that the paper's own results text refutes

Section V-C: "At $\alpha_D \geq 0.5$ ... FairPoC leads all baselines on
both datasets."
Section VII: "at all $\alpha_D \geq 0.5$ FairPoC leads or matches every
baseline on both datasets."

Section VI-B, three paragraphs earlier: "On FashionMNIST, FairPoC ...
falls below PoCL at $\alpha_D = 1.0$ (0.962 vs. 0.995)."

Table III confirms PoCL at 0.995 against FairPoC at 0.962. The claim is
false as written, and it is false in the Limitations section, which is
the one place a reviewer reads for candour. Restrict it to CIFAR-10, or
say "leads or matches except FashionMNIST at $\alpha_D = 1.0$".

This is the fourth numeric claim in this paper found to contradict its
own tables. The others were corrected earlier today. The pattern is
consistent: prose written against an earlier version of the results.
**Before submitting, re-derive every comparative claim from the current
tables rather than editing the sentences.**

### M3. PoC scores drive aggregation, which confounds the accuracy results

Algorithm 1, final line:
`theta^(t+1) <- sum_i (s_i / sum_j s_j) theta_i^(t)`

So the PoC score is not only the reward basis, it is the **aggregation
weight**. FairPoC therefore changes the model as well as the payouts,
while FedAVG aggregates uniformly.

Consequence: the accuracy column is not a clean comparison of
contribution-scoring methods. It partly measures whether PoC-weighted
aggregation beats uniform aggregation, which is a different claim and one
the paper never makes explicitly. A reviewer who notices this will ask for
an ablation holding aggregation fixed and varying only the reward rule.

This is answerable without new ideas: run FairPoC scoring with FedAVG
aggregation and report both. If accuracy is unchanged, the confound is
disposed of in one row and the paper is stronger for it. If accuracy drops,
that is worth knowing before a reviewer finds it.

### M4. Scale is below the comparison set

Ten clients. TFFL evaluates 50 and 100; BlockFL-MD2C goes to 500; the
VLDB FLCE evaluation uses 8 but is explicit that exponential schemes
force it. Section VII defends $n=10$ as "representative of private
enterprise FL", which is a fair argument, but it will not fully land
against reviewers who have just read the 100-client papers. Not fixable
before the deadline; make sure the defence is prominent rather than
buried in Limitations.

---

## MODERATE

### Mo1. Inconsistent rigour between sections

Main results use three seeds. Section VI-E (the CLP attack, the paper's
strongest contribution) uses **one**. A reviewer will ask why the headline
result is the least replicated. The limitation is stated honestly, which
helps, but three seeds on the clean and adaptive arms would remove the
objection entirely and costs a few GPU hours.

### Mo2. No dispersion reported anywhere

Tables II and III give three-seed means with no standard deviation, and
no significance testing appears in the paper. Several of the headline
margins are small: FairPoC 84.3 against FedAVG 84.2 at CIFAR-10
$\alpha_D=1.0$ is 0.1 points and is bolded as a win. Without variance a
reviewer cannot tell whether that is a result or noise, and bolding it
invites the question. Report mean +/- std; drop bolding where the margin
is inside one standard deviation.

### Mo3. The GT-Fairness ground truth is asserted, not justified

Eq. 6 sets `gt_i = 0.5 (n_i/N) + 0.5 H_i`. Why those two terms? Why equal
weights? The construction is reasonable and the non-circularity argument
is sound and valuable, but a reviewer will ask whether the conclusions are
robust to the mixing weight. A sensitivity line over gamma in {0.25, 0.5,
0.75} would close this; the code already exposes `gamma` as a parameter in
`compute_data_quality_gt`, so it is cheap.

Note also that the code normalises `gt` to sum to one and Eq. 6 does not
show it. Harmless, since Jain's index is scale-invariant, but a
reimplementer will notice.

### Mo4. Section III-B says "three on-chain functions"

The deployed contract exposes `stake`, `submitAndSettle`, `isBlacklisted`,
`balanceOf`, `slashedTotal` and three public constants. "Three functions
encoding reward distribution and adversarial defence" is defensible but
imprecise; say "three externally callable functions" or name them.

---

## MINOR

- `\TBD` is defined in the preamble and never used. Remove.
- Keyword list runs to ten entries including "gradient norm" and
  "stake-and-slash". IEEE convention is five to seven; trim.
- Section V-A's integrity remark is a `remark`, not a proposition, which
  is the right call, but the surrounding section is titled "Theoretical
  Analysis" and contains one proposition, one corollary and two remarks.
  Consider "Analysis" to avoid promising more formality than is present.
- Figure 2 caption says "Gradient norm diversity (left) vs. LOO (right)"
  while the subsection is titled "Convergence Stability of Gradient
  Divergence". Divergence and norm are used interchangeably across the
  paper. Pick one term.

---

## WHAT IS WORKING

Stated plainly, because a review that only lists faults is not useful.

1. **Section VI-E is a real contribution.** Attacking a published
   critical-period detector, with a control arm that behaves as predicted
   and an adaptive attacker that reports its own cost, is careful work.
   The fixed-scale arm producing *fewer* critical rounds than the clean
   arm is the detail that makes the result credible.
2. **The noise-floor observation is arguably stronger than the attack**
   and is currently buried in a paragraph. The detector firing with no
   adversary present needs no threat model and cannot be rebutted with
   "set tau higher". Consider leading VI-E with it.
3. **The non-circular metric is a genuine methodological contribution**,
   and the contrast with Gini-over-payouts is sharp and correct.
4. **Limitations is unusually honest** for this literature, in particular
   admitting the FashionMNIST shortfall and that scoring stays off-chain.
   Keep that; it buys goodwill that the modest contribution needs.
5. **Gas is now measured on the contract the paper describes.** Few
   papers in this space can say that.

---

## PRIORITY ORDER

1. M1 baselines (half a day, blocking)
2. M2 the false comparative claim (ten minutes, blocking)
3. Mo2 report standard deviations (an hour from existing seed CSVs)
4. M3 aggregation ablation (a few GPU hours, strongly advised)
5. Mo1 three seeds for VI-E (a few GPU hours)
6. Mo3 gamma sensitivity (an hour)
7. Minors

Items 1 and 2 must be done. Items 3 to 6 each remove a specific,
predictable reviewer objection.
