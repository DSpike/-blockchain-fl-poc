"""Global hyperparameters — override via experiments/*.yaml."""

# Federated learning
NUM_CLIENTS = 10
NUM_ROUNDS = 100
LOCAL_EPOCHS = 5
LOCAL_BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Non-IID partitioning (Dirichlet)
DIRICHLET_ALPHA = 0.5          # 0.1=extreme non-IID, 1.0=mild, 100=IID

# PoC contribution score
ALPHA_ACC = 0.5                # weight for validation accuracy term
BETA_DIV = 0.5                 # weight for gradient divergence term

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
