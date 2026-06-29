"""
BLADE-FL baseline: Blockchain-Assisted Decentralized FL (arxiv:2101.06905).
Simulated as standard FedAVG with equal weights (BLADE-FL's aggregation
is equivalent to FedAVG; its novelty is the blockchain bookkeeping layer).
"""

from baselines.fedavg import uniform_fedavg


def blade_fl_aggregate(global_model, client_models):
    return uniform_fedavg(global_model, client_models)
