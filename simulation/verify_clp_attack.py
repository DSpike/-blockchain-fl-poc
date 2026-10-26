"""
Verification experiment: is the R3T CLP detector manipulable?

R3T (Carpe Diem, IEEE TNSE 2026, DOI 10.1109/TNSE.2026.3717827) decides
which rounds are "critical learning periods" (CLPs) from a weighted sum of
squared gradient norms, their Eq. (4):

    FGN(t) = -eta * sum_n omega_n * ||g_n||^2 ,   omega_n = e_n / sum_m e_m

and classifies round t as a CLP when the RELATIVE change exceeds a
threshold, their Eq. (5):

    ( FGN(t) - FGN(t-1) ) / FGN(t-1)  >=  tau_CLP

CLP rounds carry the bonus multiplier h(t) = 1 + vartheta / ln(2t); non-CLP
rounds carry h(t) = 1.

HYPOTHESIS UNDER TEST
    Because the test is on the RELATIVE change, a client that ramps its
    update scale each round can hold rounds in CLP state far beyond the
    true critical window, extending the high-bonus schedule and draining
    the reward budget. Median-clipping the norms before forming FGN should
    remove the effect.

    This is derived from their published equations and is NOT an
    established result. If the ramp arm does not separate from the clean
    arm, the hypothesis is wrong and the paper claim must be dropped.

ARMS
    clean    no attacker
    fixed    attacker scales by a constant factor (control: a constant
             scale shifts FGN once and should NOT extend the CLP window)
    ramp     attacker scales by a geometrically growing factor
    ramp_mc  ramp attack, FGN formed from median-clipped norms

NOTE ON THE PROXY
    R3T forms FGN from mini-batch gradients. This harness has the
    parameter delta (local minus global), which is -lr times the
    accumulated gradient. We use the squared delta norm as the proxy and
    state so in the paper. The attack argument depends only on the
    attacker's ability to inflate its own term, which holds either way.

Usage:
    python simulation/verify_clp_attack.py --rounds 40 --alpha 0.1
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
import math
import time
import numpy as np
import torch

from fl.client import FLClient
from fl.aggregator import weighted_fedavg
from fl.contribution import flatten_params
from evaluation.metrics import accuracy
from simulation.run_adversarial import _build_loaders, _build_model
from config import NUM_CLIENTS, LOCAL_EPOCHS, LOCAL_BATCH_SIZE, LEARNING_RATE

ATTACKER_IDX = 0
TAU_CLP = 0.01      # R3T Table II, CIFAR-10
VARTHETA = 1.0      # bonus unit coefficient; R3T leaves this free
THETA = 1.0         # per-client capability index, held uniform here


def h_bonus(t, is_clp):
    """R3T Eq. (2): time-aware bonus unit function."""
    if is_clp and t >= 1:
        return 1.0 + VARTHETA / math.log(2.0 * t)
    return 1.0


def fgn(norms_sq, omega, clip):
    """R3T Eq. (4), optionally with median-clipped norms."""
    v = np.asarray(norms_sq, dtype=float)
    if clip == "median":
        v = np.minimum(v, float(np.median(v)))
    return float(np.sum(np.asarray(omega, dtype=float) * v))


def run_arm(arm, cfg, loaders, val_loader, sizes, device):
    n = cfg["num_clients"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    global_model = _build_model(cfg["dataset"]).to(device)
    clients = [FLClient(i, global_model, loaders[i], lr=cfg["lr"],
                        local_epochs=cfg["local_epochs"], device=device)
               for i in range(n)]

    # omega_n: R3T weights by contracted effort, and effort is the amount of
    # contributed data, so we use the client partition sizes.
    omega = np.asarray(sizes, dtype=float)
    omega = omega / omega.sum()

    clip = "median" if arm.endswith("_mc") else "max"
    rows = []
    prev_fgn = None
    cum_bonus = 0.0
    cum_bonus_attacker = 0.0

    for rnd in range(1, cfg["num_rounds"] + 1):
        for c in clients:
            c.set_global_weights(global_model.state_dict())
        local_models = [c.train() for c in clients]

        # ---- attack injection on the parameter delta ----
        if arm.startswith("fixed"):
            s = cfg["scale"]
        elif arm.startswith("ramp"):
            s = cfg["scale"] ** (1.0 + cfg["ramp"] * (rnd - 1))
        else:
            s = 1.0
        if s != 1.0:
            with torch.no_grad():
                for p_l, p_g in zip(local_models[ATTACKER_IDX].parameters(),
                                    global_model.parameters()):
                    p_l.data = p_g.data + (p_l.data - p_g.data) * s

        # ---- squared update norms ----
        gvec = flatten_params(global_model)
        norms_sq = []
        for m in local_models:
            d = flatten_params(m) - gvec
            norms_sq.append(float((d.norm() ** 2).item()))

        # ---- R3T CLP detection ----
        f = fgn(norms_sq, omega, clip)
        if prev_fgn is None or prev_fgn == 0.0:
            is_clp = True          # round 1 is inside the window by construction
            rel = float("nan")
        else:
            rel = (f - prev_fgn) / abs(prev_fgn)
            is_clp = bool(rel >= TAU_CLP)
        prev_fgn = f

        # ---- bonus schedule ----
        h = h_bonus(rnd, is_clp)
        bonus_n = THETA * h * omega          # per-client bonus this round
        cum_bonus += float(bonus_n.sum())
        cum_bonus_attacker += float(bonus_n[ATTACKER_IDX])

        # ---- aggregate and continue training ----
        # signature is weighted_fedavg(global_model, client_models, weights)
        global_model = weighted_fedavg(global_model, local_models,
                                       omega.tolist())
        acc = accuracy(global_model, val_loader, device)

        rows.append(dict(arm=arm, round=rnd, scale=s, fgn=f, rel_change=rel,
                         is_clp=int(is_clp), h=h, cum_bonus=cum_bonus,
                         cum_bonus_attacker=cum_bonus_attacker, acc=acc))
        rel_show = 0.0 if rel != rel else rel
        print("  [%-8s] r%02d  s=%8.2f  FGN=%10.4g  rel=%8.4f  CLP=%d  "
              "h=%.3f  budget=%8.3f  acc=%.4f"
              % (arm, rnd, s, f, rel_show, is_clp, h, cum_bonus, acc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--clients", type=int, default=NUM_CLIENTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="base scale factor for the attacker")
    ap.add_argument("--ramp", type=float, default=0.15,
                    help="geometric growth of the exponent per round")
    ap.add_argument("--out", default="results/clp_attack_verification.csv")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # build_client_loaders returns 5 values:
    #   client_loaders, val_loader, partitions, full_val_loader, targets
    loaders, val_loader, partitions, full_val_loader, targets = _build_loaders(
        a.dataset, a.alpha, a.clients, LOCAL_BATCH_SIZE)
    sizes = [len(p) for p in partitions]
    print("partition sizes:", sizes)

    cfg = dict(dataset=a.dataset, num_clients=a.clients, num_rounds=a.rounds,
               seed=a.seed, lr=LEARNING_RATE, local_epochs=LOCAL_EPOCHS,
               scale=a.scale, ramp=a.ramp)

    all_rows = []
    for arm in ["clean", "fixed", "ramp", "ramp_mc"]:
        print("=" * 72)
        print("ARM:", arm)
        t0 = time.time()
        all_rows += run_arm(arm, cfg, loaders, val_loader, sizes, device)
        print("  (%.1f min)" % ((time.time() - t0) / 60.0))

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # ---- verdict ----
    print("=" * 72)
    print("SUMMARY")
    print("%-9s %9s %11s %13s %14s %8s"
          % ("arm", "CLP rnds", "last CLP", "total bonus", "attacker sh.", "acc"))
    base = None
    for arm in ["clean", "fixed", "ramp", "ramp_mc"]:
        r = [x for x in all_rows if x["arm"] == arm]
        nclp = sum(x["is_clp"] for x in r)
        clp_rounds = [x["round"] for x in r if x["is_clp"]]
        last = max(clp_rounds) if clp_rounds else 0
        tot = r[-1]["cum_bonus"]
        shr = r[-1]["cum_bonus_attacker"] / tot if tot else 0.0
        if arm == "clean":
            base = (nclp, tot)
        print("%-9s %9d %11d %13.3f %14.4f %8.4f"
              % (arm, nclp, last, tot, shr, r[-1]["acc"]))
    print("")

    ramp = [x for x in all_rows if x["arm"] == "ramp"]
    ramp_mc = [x for x in all_rows if x["arm"] == "ramp_mc"]
    nclp_ramp = sum(x["is_clp"] for x in ramp)
    nclp_mc = sum(x["is_clp"] for x in ramp_mc)
    drain = ramp[-1]["cum_bonus"] / base[1] if base[1] else 0.0
    drain_mc = ramp_mc[-1]["cum_bonus"] / base[1] if base[1] else 0.0

    print("CLP rounds:  clean=%d   ramp=%d   ramp_mc=%d"
          % (base[0], nclp_ramp, nclp_mc))
    print("Budget drain vs clean:  ramp=%.3fx   ramp_mc=%.3fx"
          % (drain, drain_mc))
    print("")
    if nclp_ramp > base[0] and drain > 1.05:
        print("VERDICT: hypothesis SUPPORTED. The ramp attack extends the CLP")
        print("         window and inflates the bonus budget.")
        if nclp_mc < nclp_ramp:
            print("         Median-clip reduces the extension, so the defence")
            print("         carries over to this setting.")
        else:
            print("         WARNING: median-clip did NOT reduce the extension.")
            print("         The defence claim does not carry over as written.")
    else:
        print("VERDICT: hypothesis NOT SUPPORTED. Do not write this claim.")
        print("         Consider: larger --ramp, lower --tau, or the")
        print("         possibility that omega_n damps the attacker's term.")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
