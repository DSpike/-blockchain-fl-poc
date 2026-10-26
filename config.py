"""Global hyperparameters — override via experiments/*.yaml."""

# Federated learning
NUM_CLIENTS = 10
NUM_ROUNDS = 50          # paper uses 50 rounds
LOCAL_EPOCHS = 3         # paper uses 3 local epochs
LOCAL_BATCH_SIZE = 32    # paper uses batch 32
LEARNING_RATE = 1e-3     # paper uses lr 1e-3

# Non-IID partitioning (Dirichlet)
DIRICHLET_ALPHA = 0.5          # 0.1=extreme non-IID, 1.0=mild, 100=IID

# PoC contribution score (paper default: alpha=0.7, beta=0.3)
ALPHA_ACC = 0.7                # weight for validation accuracy term
BETA_DIV  = 0.3                # weight for gradient norm term

# Byzantine simulation
BYZANTINE_FRACTION = 0.2       # fraction of malicious clients per round
BYZANTINE_ATTACK = "random"    # "random" | "sign_flip" | "scaling"

# Blockchain / Ganache
GANACHE_URL = "http://127.0.0.1:8545"
GAS_LIMIT = 3_000_000

# Miners
NUM_MINERS = 2
CONSENSUS_THRESHOLD = 0.67     # 67% agreement required

# ERC20 reward pool (tokens per round)
REWARD_PER_ROUND = 100

# Reproducibility
SEED = 42
