"""
Non-IID-aware Proof of Contribution score.

contribution_i = alpha * val_acc_i + beta * grad_div_i

  val_acc_i  : accuracy of client i's model on shared validation set
  grad_div_i : 1 - cosine_similarity(grad_i, mean_grad)
               captures how unique client i's update direction is,
               which correlates with rare-class data under non-IID

This is the core novelty of the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np


def flatten_params(model):
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])


def gradient_magnitude(local_grad: torch.Tensor) -> float:
    """
    L2 norm of the client's parameter update (unnormalised).
    Clients with more data and more diverse classes produce larger updates.
    Normalisation across clients is done in compute_contribution_scores.
    """
    return local_grad.norm().item()


def gradient_inner_product(local_grad: torch.Tensor, mean_grad: torch.Tensor) -> float:
    """
    Projection of client gradient onto the mean gradient direction:
        c_i = <Δw_i, Δw_avg> / ||Δw_avg||

    Unlike raw gradient norm, this is scale-invariant across datasets and
    model architectures: a client with misaligned (harmful) update receives
    a low or negative score regardless of its magnitude, while a client
    with a large, well-aligned update is rewarded for both magnitude and
    direction. Negative values indicate updates that hurt the consensus and
    are clipped to zero before scoring.
    """
    mean_norm = mean_grad.norm().item() + 1e-9
    return torch.dot(local_grad.cpu(), mean_grad.cpu()).item() / mean_norm


def gradient_divergence(local_grad: torch.Tensor, mean_grad: torch.Tensor) -> float:
    """
    1 - cosine_similarity(local_grad, mean_grad), normalised to [0,1].
    NOTE: kept for reference only. Under non-IID conditions this signal is
    INVERSELY correlated with data diversity — clients with more diverse data
    produce gradients closer to the mean, receiving lower scores. Use
    gradient_magnitude instead.
    """
    cos_sim = F.cosine_similarity(local_grad.unsqueeze(0),
                                  mean_grad.unsqueeze(0)).item()
    return (1.0 - cos_sim) / 2.0


def evaluate_accuracy(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def compute_contribution_scores(client_models, global_model, val_loader,
                                 device, alpha=0.5, beta=0.5,
                                 grad_norm_clip="max", grad_signal="norm"):
    """
    Args:
        client_models   : list of trained local models (one per client)
        global_model    : current global model (before aggregation)
        val_loader      : shared validation DataLoader
        device          : torch device
        alpha, beta     : weighting coefficients (alpha + beta = 1 recommended)
        grad_norm_clip  : "max"    — normalise by max (default)
                          "median" — clip to median before normalising
                                     (limits gradient inflation attackers)
        grad_signal     : "norm"          — raw gradient L2 norm (original)
                          "inner_product" — projection onto mean gradient:
                                           <Δw_i, Δw_avg> / ||Δw_avg||
                                           scale-invariant across datasets/models

    Returns:
        scores : np.array of shape (num_clients,), normalised to sum=1
        raw    : dict with val_acc, grad_norm, grad_div per client (for logging)
    """
    global_vec = flatten_params(global_model)
    grads = [flatten_params(m) - global_vec for m in client_models]
    mean_grad = torch.stack(grads).mean(dim=0)

    val_accs, grad_divs = [], []
    for i, model in enumerate(client_models):
        val_accs.append(evaluate_accuracy(model, val_loader, device))
        grad_divs.append(gradient_divergence(grads[i], mean_grad))

    val_accs  = np.array(val_accs)
    grad_divs = np.array(grad_divs)

    if grad_signal == "inner_product":
        raw_ip = np.array([gradient_inner_product(g, mean_grad) for g in grads])
        # clip negatives (harmful updates get zero contribution credit)
        raw_ip = np.maximum(raw_ip, 0.0)
        if grad_norm_clip == "median":
            med = np.median(raw_ip) + 1e-9
            grad_norm = np.minimum(raw_ip, med) / med
        else:
            grad_norm = raw_ip / (raw_ip.max() + 1e-9)
    else:
        norms = np.array([gradient_magnitude(g) for g in grads])
        if grad_norm_clip == "median":
            med = np.median(norms) + 1e-9
            grad_norm = np.minimum(norms, med) / med
        else:
            grad_norm = norms / (norms.max() + 1e-9)

    raw_scores = alpha * val_accs + beta * grad_norm
    normalised = raw_scores / (raw_scores.sum() + 1e-9)

    return normalised, {"val_acc": val_accs, "grad_div": grad_divs,
                        "grad_norm": grad_norm, "raw": raw_scores}


def compute_loo_contributions(local_models, global_model, val_loader, device):
    """
    Leave-One-Out (LOO) contribution for each client.

    LOO_i = Acc(aggregate ALL clients) - Acc(aggregate ALL clients EXCEPT i)

    This is an objective, model-independent ground truth for client contribution:
    it measures how much the global model quality degrades when client i is removed.
    Positive LOO_i means client i genuinely improves the global model.

    Returns np.array of shape (n_clients,). Values can be negative (harmful client).
    """
    import copy
    from fl.aggregator import weighted_fedavg

    n = len(local_models)
    uniform = np.ones(n) / n

    # Accuracy of full aggregation (all n clients, uniform weights)
    full_model = copy.deepcopy(global_model)
    weighted_fedavg(full_model, local_models, uniform)
    full_acc = evaluate_accuracy(full_model, val_loader, device)

    loo = np.zeros(n)
    for i in range(n):
        others = [m for j, m in enumerate(local_models) if j != i]
        w = np.ones(n - 1) / (n - 1)
        loo_model = copy.deepcopy(global_model)
        weighted_fedavg(loo_model, others, w)
        loo[i] = full_acc - evaluate_accuracy(loo_model, val_loader, device)

    return loo


def compute_data_quality_gt(partitions, all_targets, num_classes=10, gamma=0.5):
    """
    Independent contribution ground truth derived purely from client data partitions.

    gt_i = gamma * (n_i / sum_j n_j)  +  (1-gamma) * H_i

    where H_i = normalised Shannon entropy of client i's class distribution.
    High H_i = client holds diverse classes (more valuable under non-IID).
    High n_i = client holds more data (more valuable universally).

    Returns a normalised np.array (sums to 1), shape (num_clients,).
    This is completely independent of any model or scoring metric.
    """
    n_clients = len(partitions)
    sizes     = np.array([len(p) for p in partitions], dtype=float)
    vol_frac  = sizes / sizes.sum()

    entropy = np.zeros(n_clients)
    for i, idxs in enumerate(partitions):
        labels = all_targets[np.array(idxs)]
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log(p)) / np.log(num_classes)  # in [0, 1]
        entropy[i] = H

    gt = gamma * vol_frac + (1.0 - gamma) * entropy
    gt = gt / gt.sum()
    return gt


def fltrust_scores(local_models, global_model, root_loader, device,
                   lr=1e-3, local_epochs=1):
    """
    FLTrust (Cao et al., 2020) trust scores.

    The aggregator trains one local model on its clean root dataset (here: the
    shared validation loader) for `local_epochs` epochs, producing a trusted
    update direction.  Each client's cosine similarity with that direction
    becomes its trust score.  Clients whose updates diverge from the server's
    direction (cos < 0) receive 0 trust, providing Byzantine robustness.

    Returns:
        trust_scores : np.array (n,), non-negative, sums to 1.
        server_norm  : L2 norm of the server update (for clipping reference).
    """
    import copy
    import torch.nn as nn, torch.optim as optim

    server_model = copy.deepcopy(global_model).to(device)
    server_model.train()
    opt       = optim.SGD(server_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x, y in root_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            criterion(server_model(x), y).backward()
            opt.step()

    server_grad = flatten_params(server_model) - flatten_params(global_model)
    server_norm = server_grad.norm().item() + 1e-9
    global_vec  = flatten_params(global_model)

    trust = []
    for m in local_models:
        cg  = flatten_params(m) - global_vec
        cos = F.cosine_similarity(cg.unsqueeze(0), server_grad.unsqueeze(0)).item()
        trust.append(max(0.0, cos))

    ts    = np.array(trust)
    total = ts.sum()
    if total < 1e-9:
        ts = np.ones(len(local_models)) / len(local_models)
    else:
        ts = ts / total
    return ts, server_norm


def krum_select(local_models, global_model, n_byzantine=2, multi_k=None):
    """
    Multi-Krum (Blanchard et al., 2017) Byzantine-robust client selection.

    For each client i, computes the sum of squared L2 distances to its
    (n - n_byzantine - 2) nearest neighbours in parameter space.  The
    `multi_k` clients with the lowest scores are selected for aggregation;
    higher-scoring clients (whose updates are distant from the cluster) are
    excluded as likely Byzantine.

    Args:
        local_models : list of trained local models.
        global_model : current global model (provides reference parameter space).
        n_byzantine  : number of suspected Byzantine clients (default 2 = 20 % of 10).
        multi_k      : number of clients to select (default n - n_byzantine).

    Returns:
        keep : boolean np.array of shape (n,); True = selected for aggregation.
    """
    n   = len(local_models)
    m   = multi_k if multi_k is not None else max(1, n - n_byzantine)
    gv  = flatten_params(global_model)
    vecs = [flatten_params(mod) - gv for mod in local_models]

    # pairwise squared L2 distances in update space
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = (vecs[i] - vecs[j]).norm().item() ** 2
            dists[i, j] = d
            dists[j, i] = d

    k_nn = max(1, min(n - n_byzantine - 2, n - 1))
    scores = np.zeros(n)
    for i in range(n):
        row       = np.sort(dists[i])
        scores[i] = row[1: k_nn + 1].sum()   # skip self (index 0 = 0.0)

    selected = np.argsort(scores)[:m]
    keep     = np.zeros(n, dtype=bool)
    keep[selected] = True
    return keep


def filter_byzantine(scores, raw, threshold_std=2.0):
    """
    Exclude clients whose raw contribution score is > threshold_std below mean.
    Returns boolean mask (True = keep).
    """
    raw_scores = raw["raw"]
    cutoff = raw_scores.mean() - threshold_std * raw_scores.std()
    return raw_scores >= cutoff


def tmc_shapley(local_models, global_model, val_loader, device,
                T=20, truncation_frac=0.99):
    """
    Truncated Monte Carlo Shapley (TMC-Shapley) for FL contribution scoring.

    Each round:
      - Sample T random permutations of clients.
      - For each permutation, evaluate marginal contribution of each client
        by forming cumulative FedAVG coalitions and measuring validation accuracy.
      - Use truncation: stop adding clients when accuracy exceeds truncation_frac
        of the full-coalition accuracy (saves evaluations near convergence).

    Returns:
        sv_normalised : np.array shape (n,), non-negative, sums to 1.
                        Clients with negative Shapley values receive 0 reward.

    Reference: Ghorbani & Zou (2019), Data Shapley.
    """
    import copy
    from fl.aggregator import weighted_fedavg

    n         = len(local_models)
    sv        = np.zeros(n)
    baseline  = evaluate_accuracy(global_model, val_loader, device)

    # full-coalition accuracy (all clients, uniform weights)
    full_model = copy.deepcopy(global_model)
    uniform    = np.ones(n) / n
    weighted_fedavg(full_model, local_models, uniform)
    full_acc   = evaluate_accuracy(full_model, val_loader, device)

    trunc_thresh = truncation_frac * full_acc

    for _ in range(T):
        perm     = np.random.permutation(n)
        prev_acc = baseline
        for k, i in enumerate(perm):
            coalition = list(perm[: k + 1])
            tmp = copy.deepcopy(global_model)
            w   = np.ones(len(coalition)) / len(coalition)
            weighted_fedavg(tmp, [local_models[j] for j in coalition], w)
            curr_acc = evaluate_accuracy(tmp, val_loader, device)
            sv[i]    += curr_acc - prev_acc
            prev_acc  = curr_acc
            if curr_acc >= trunc_thresh:
                break

    sv /= T
    # reward only positive contributions; negative = harmful client
    sv_pos = np.maximum(sv, 0.0)
    total  = sv_pos.sum()
    if total < 1e-9:
        sv_pos = np.ones(n) / n   # fallback: uniform
    else:
        sv_pos = sv_pos / total
    return sv_pos
