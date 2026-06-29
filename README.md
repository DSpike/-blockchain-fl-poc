# Blockchain-FL: Non-IID-Aware Proof of Contribution

> A decentralized blockchain-enabled federated learning framework with a heterogeneity-aware Proof of Contribution (PoC) consensus mechanism.

## Novelty

Existing PoC mechanisms (e.g., PoCL, arxiv:2407.13018) score clients by **validation accuracy alone**, assuming IID data. Under non-IID distributions, clients holding rare classes contribute unique information but are penalised by low accuracy scores.

We propose a two-term contribution score:

```
contribution_i = α × val_acc_i + β × grad_div_i
```

where `grad_div_i = 1 - cosine_similarity(Δw_i, mean(Δw))` captures **gradient direction uniqueness** — a proxy for data heterogeneity. Clients with rare classes produce divergent gradients and are rewarded fairly.

## Project Structure

```
data/           CIFAR-10 loader + Dirichlet non-IID partitioning
models/         ResNet-8 (fast CIFAR-10 benchmark model)
fl/             Client training, weighted FedAVG, contribution scoring
blockchain/     Solidity contracts + Web3.py interface + Ganache deploy
simulation/     Training loops (FL-only and full blockchain)
baselines/      FedAVG, PoCL, BLADE-FL
evaluation/     Metrics (accuracy, fairness index, Byzantine resilience)
experiments/    YAML configs for each paper experiment
```

## Quick Start

```bash
pip install -r requirements.txt

# FL only (no blockchain — fastest way to test PoC mechanism)
python simulation/run_federated.py --config experiments/exp1_iid_vs_noniid.yaml

# Full system (requires Ganache)
npx ganache
python blockchain/deploy.py
python simulation/run_blockchain_fl.py
```

## Experiments

| # | Config | What it shows |
|---|--------|---------------|
| 1 | `exp1_iid_vs_noniid.yaml` | PoC fairness advantage under α=0.1 non-IID |
| 2 | `exp2_dirichlet_alpha.yaml` | Fairness index vs heterogeneity level (sweep α) |
| 3 | `exp3_byzantine.yaml` | Resilience under 20–30% malicious clients |

## Baselines

| Method | Description |
|--------|-------------|
| FedAVG | Centralized, uniform weights |
| PoCL | PoC with accuracy-only scoring (our main competitor) |
| BLADE-FL | Blockchain-assisted FL, equal weights |

## Citation (closest prior work)

```
@article{pocl2024,
  title={Proof-of-Collaborative-Learning: A Multi-winner Federated Learning Consensus Algorithm},
  author={...},
  journal={arXiv:2407.13018},
  year={2024}
}
```
