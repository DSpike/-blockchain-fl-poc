# Does the FLCE impossibility result hold?

Working sketch, 2026-11-12. Written against the open question posed in:

> Chen, Li, Li, Wang. *Contributions Estimation in Federated Learning: A
> Comprehensive Experimental Evaluation.* PVLDB 17(8):2077-2090, 2024.
> Section 7, Prospect (2), p.2088:
>
> "No contribution estimation scheme exists that can fulfill the criteria of
> effectiveness, robustness, and efficiency simultaneously. It would be
> beneficial to either explore a more balanced trade-off among these three
> aspects **or prove that it is not feasible to simultaneously meet all three
> requirements in FLCE.**"

Purpose of this note: decide whether there is a theorem here, or only a
reframing. Verdict first, argument after.

---

## Verdict

**The obvious impossibility is already known and is not publishable.**
**A second, sharper one looks genuinely open and is the one to pursue.**

Read sections 1 and 2 to see why the obvious route is dead; section 3 is the
candidate result; section 5 lists what would kill it.

---

## 1. Setup and definitions

Following the VLDB decomposition, an FLCE method is a pair (v, phi):

- a **data utility metric** v: 2^N -> R, v(empty) = 0
- a **contribution estimation scheme** phi_i built from evaluations of v

All four schemes they evaluate are linear functionals of v:

    phi_i = sum over S of w_i(S) * v(S)

| scheme     | weights w_i                                      | nonzero terms |
|------------|--------------------------------------------------|---------------|
| Individual | w_i({i}) = 1                                     | 1             |
| LeaveOneOut| w_i(N) = 1, w_i(N\{i}) = -1                      | 2             |
| Shapley    | +/- c_{|S|} over all S subset of N\{i}           | 2^n           |
| LeastCore  | solution of an LP over all 2^n coalitions        | 2^n           |

The three criteria, stated so they can be argued about:

- **Efficiency** E(phi) = number of distinct coalitions whose utility must be
  evaluated. (VLDB measure it as wall-clock, but it tracks this count; their
  Finding 3 says so explicitly.)
- **Effectiveness** Eff(v, phi) = the estimate orders clients by their true
  importance in the grand coalition. VLDB cannot measure this directly
  ("obtaining the ground truth of clients' contributions is not feasible")
  and substitute the client-removal curve.
- **Robustness** Rob(v, phi) = for every adverse behaviour a available to a
  client, the relative contribution change (phi_adv - phi_orig)/|phi_orig|
  is <= 0. Their adverse set: data replication, random data generation,
  low-quality data, label flipping.

---

## 2. The obvious impossibility is Shapley uniqueness. Do not publish it.

The tempting argument runs: effectiveness demands the Shapley axioms (group
rationality, symmetry, null player, additivity); Shapley's 1953 uniqueness
theorem says the only functional satisfying all four is the Shapley value;
the Shapley value needs 2^n evaluations; therefore effectiveness and
efficiency are incompatible.

Every step is true. It is also **entirely known** - it is Shapley uniqueness
plus a complexity count, and VLDB's own Table 1 displays it as a table of
ticks and crosses. Writing it up as a theorem would be restating 1953 in new
notation, and it says nothing at all about robustness, which is the criterion
that actually interests us.

A second dead end worth recording: trying to make robustness scale with the
number of sampled coalitions. The intuition is that averaging over more
coalitions dilutes a manipulation. **It does not.** An adverse client changes
v(S) for *every* coalition S containing itself, so its manipulation is present
in every term of its own estimate and averaging does not attenuate it. This is
consistent with VLDB's finding that Individual (one coalition) and Shapley
(2^n coalitions) are *both* robust while LeaveOneOut (two coalitions) is not -
robustness is not monotone in the coalition count, so no argument of that
shape can work.

---

## 3. The candidate: effectiveness under non-IID and robustness to synthesis
   are in tension, and the only escape is expensive

### 3.1 The claim

> **Conjecture.** No data utility metric v can simultaneously satisfy:
>
> **(a) Reference-free.** v is computable without a held-out test set drawn
>     from the task distribution.
> **(b) Heterogeneity-effective.** Under non-IID partitions, v assigns higher
>     utility to a coalition carrying complementary (rare, distribution-
>     extending) data than to one carrying redundant data of equal volume.
> **(c) Synthesis-robust.** A client cannot raise v by injecting data it
>     generated rather than collected.
>
> Any two are achievable; all three are not.

### 3.2 Why (a) and (b) force a diversity-like functional

Without a reference distribution, v has access only to the submitted data or
updates. "Complementary" can then only mean *distributionally distinct from
what the other clients submitted* - there is no other information in the
system that distinguishes useful novelty from noise. So v must be an
increasing function of some dispersion or novelty functional of the submitted
data relative to the rest.

VLDB's own taxonomy is the evidence: every test-set-independent metric they
survey is of this form. Volume is a Gram determinant (dispersion in feature
space). RobustVolume is Volume with replication discounted. CosineGradient
measures deviation of a coalition's update direction. InformationGain measures
reduction in parameter entropy. FairPoC's normalised gradient norm is
magnitude of the update, which is the same family.

### 3.3 Why a diversity-like functional cannot satisfy (c)

Synthetic data drawn from a distribution *wider* than the client's true data
increases dispersion. Since v is increasing in dispersion and has no reference
against which to check whether the added spread corresponds to anything real,
v rises. The attacker needs no knowledge of other clients, only of v's
monotonicity direction.

The empirical record in VLDB is exactly this, and is unusually clean:

- Volume is inflatable by **replication**; RobustVolume was introduced to
  patch that one hole (section 3.2.1).
- RobustVolume then shows "weak robustness against random generation"
  (section 6.5.2) - it closed replication and left synthesis open.
- DataQuantity "does not possess the capability to identify random generation
  and label-flipping behaviors, because it does not consider data quality."
- CosineGradient "can identify most label-flipping behaviors, but its
  robustness against random generation is low."

Four metrics, four different failures, all against synthesis or quality.
And their section 3.3 closes: "when there are a moderate number of malicious
clients or the data distribution is too complex, **no existing metrics can
reflect the data utility well**." Finding 4: "**no existing metrics can
completely replace Accuracy** in cases of heavy adversity like significant
label flipping."

That is a survey reporting a pattern it does not explain. The conjecture
explains it.

### 3.4 Why the escape is expensive, which brings in efficiency

The escape from the trap is to drop (a): use a reference. With a held-out test
set, Accuracy distinguishes synthetic data from real, because synthetic data
does not improve test performance. VLDB confirm Accuracy is the only metric
robust across their whole adverse set.

But ModelPerformance requires **training a model per coalition**. Combined
with the effectiveness pressure toward many coalitions (section 2), the
reference-based escape is the expensive corner. Their Table 3 makes the cost
concrete: on Tic-Tac-Toe, Accuracy costs 8.26e2 s against DataQuantity's
1.09e-2 s - roughly five orders of magnitude.

So the three-way statement is:

| want | pay |
|---|---|
| effectiveness + robustness | a reference, hence retraining per coalition: efficiency dies |
| effectiveness + efficiency | reference-free diversity metric: synthesis-robustness dies |
| robustness + efficiency | Individual on DataQuantity: effectiveness under non-IID dies |

Each cell is instantiated by a real method in their evaluation, which is a
good sign the partition is the right one.

---

## 4. How FairPoC instantiates it

Normalised gradient norm is reference-free (a) and heterogeneity-effective
(b): it does not vanish at convergence because local gradients stay apart
under non-IID, which is the measured stability result. The conjecture then
*predicts* that it must fail (c) - and GradScale is exactly that failure,
measured at 1.42x over-reward.

Median-clip does not refute the conjecture. It bounds the attacker's leverage
without restoring (c): clipping caps how far synthesis can carry you, it does
not let the metric tell synthesis from complementarity. That is the honest
reading of 1.42x -> 0.78x, and it should be written that way rather than as a
fix.

The R3T CLP detector is a second instantiation. FGN is reference-free and
magnitude-based, so it sits in the same corner, and the measured attack
(2 -> 26 CLP rounds, 1.18x budget drain, 9.3x peak inflation) is the same
failure appearing in a scheduling decision rather than a payment.

Two systems, one mechanism, predicted rather than observed. That is what makes
this a contribution instead of a pair of anecdotes.

---

## 5. What is missing, and what would kill it

**Missing - these are the real work:**

1. A formal definition of "complementary" that does not smuggle in a
   reference. This is the crux. If complementarity can only be defined
   relative to a task distribution, step 3.2 may be circular rather than
   proved. **Attack this first; everything rests on it.**
2. A proof, rather than an appeal to VLDB's taxonomy, that every reference-
   free metric is increasing in some dispersion functional. As written, 3.2
   is an induction over the metrics that happen to exist.
3. A precise adversary model. "Data it generated rather than collected" needs
   to be a formal class, or (c) is not a mathematical condition.

**Falsifiers - any of these ends it:**

- A reference-free metric that is provably synthesis-robust and still
  heterogeneity-effective. Check InformationGain seriously: VLDB **excluded it
  from their experiments** (section 6.1.2, it needs a Bayesian layer), and
  section 3.2.2 claims it "remains effective in measuring data utility" under
  malicious clients because poisoned data raises parameter uncertainty. If
  that claim survives testing, the conjecture is false as stated. **This is
  the single biggest risk and it is cheap to test.**
- Peer prediction. KFCA (arXiv 2605.04747) elicits truthfully without ground
  truth under an honest majority. If an honest majority counts as a reference,
  the conjecture needs restating; if it does not, KFCA may be a counterexample.
- A cheap reference. If a coalition's utility can be estimated without
  retraining to sufficient fidelity, the expensive corner stops being
  expensive and the three-way tension collapses to a two-way one.

---

## 6. Assessment

The conjecture is **not obviously true and not obviously known**, which is the
right starting position. It explains four separate empirical failures reported
in a VLDB paper that does not explain them, it predicts FairPoC's own GradScale
result rather than rationalising it after the fact, and it answers a question
that paper poses in print.

It is also **unproven, and gap 1 may be fatal.** If complementarity cannot be
defined reference-free without circularity, this degrades from a theorem to a
well-organised observation - still worth a section, not worth a paper.

**Next step: spend a day on gap 1 and on the InformationGain falsifier.**
Those two decide it. Everything else in this note is downstream.
