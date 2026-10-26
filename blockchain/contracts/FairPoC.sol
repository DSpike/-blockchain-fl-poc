// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

/**
 * FairPoC: Blockchain-enforced FL incentive with adversarial stake-and-slash.
 *
 * Novel contribution: the PoC score serves dual purpose —
 *   (1) proportional reward for honest clients
 *   (2) on-chain evidence for autonomous adversary slashing
 *
 * A client whose score falls below SLASH_THRESHOLD_E4 (as a fraction of the
 * equal-share baseline 1/n) loses its staked deposit; the slash pool is
 * redistributed to honest clients in the same round.  No trusted third party
 * decides who is an attacker — the contract enforces the rule.
 */
contract FairPoC {

    // ------------------------------------------------------------------ state
    address public immutable aggregator;
    uint256 public constant STAKE_AMOUNT    = 1000;   // tokens required per round
    uint256 public constant SLASH_THRESHOLD = 3000;   // 3000/10000 = 30% of equal share
    uint256 public constant SCALE           = 10000;  // fixed-point denominator

    mapping(address => uint256) public balances;
    mapping(address => uint256) public stakes;
    mapping(address => bool)    public blacklisted;
    mapping(address => uint256) public totalSlashed;  // cumulative slash per client
    uint256 public immutable totalSupply;

    // ----------------------------------------------------------------- events
    event Staked(address indexed client, uint256 amount, uint256 round);
    event Rewarded(address indexed client, uint256 amount, uint256 round);
    event Slashed(address indexed client, uint256 amount, uint256 round,
                  uint256 scoreE4, uint256 thresholdE4);
    event Blacklisted(address indexed client, uint256 round);
    event ScoreSubmitted(address indexed client, uint256 round, uint256 scoreE4);

    // --------------------------------------------------------------- modifier
    modifier onlyAggregator() {
        require(msg.sender == aggregator, "Only aggregator");
        _;
    }

    // ------------------------------------------------------------ constructor
    constructor(uint256 initialSupply) {
        aggregator = msg.sender;
        totalSupply = initialSupply;
        balances[aggregator] = initialSupply;
    }

    // ----------------------------------------------------------------- staking
    /**
     * Client calls this each round to lock STAKE_AMOUNT tokens.
     * Blacklisted clients cannot stake and are permanently excluded.
     */
    function stake(uint256 round) external {
        require(!blacklisted[msg.sender], "Client is blacklisted");
        require(balances[msg.sender] >= STAKE_AMOUNT, "Insufficient balance to stake");
        balances[msg.sender] -= STAKE_AMOUNT;
        stakes[msg.sender]   += STAKE_AMOUNT;
        emit Staked(msg.sender, STAKE_AMOUNT, round);
    }

    // -------------------------------------------------------- submit & settle

    // Internal helper: classify clients, slash bad actors, return totals.
    // Extracted to avoid EVM stack-depth limit in submitAndSettle.
    function _classifyAndSlash(
        address[] calldata clients,
        uint256[] calldata scoresE4,
        uint256            thresholdE4,
        uint256            round,
        bool[]  memory     honest
    ) internal returns (uint256 slashPool, uint256 honestTotalScore) {
        for (uint i = 0; i < clients.length; i++) {
            emit ScoreSubmitted(clients[i], round, scoresE4[i]);
            if (scoresE4[i] < thresholdE4) {
                uint256 penalty = stakes[clients[i]];
                stakes[clients[i]]        = 0;
                totalSlashed[clients[i]] += penalty;
                blacklisted[clients[i]]   = true;
                slashPool                += penalty;
                honest[i]                 = false;
                emit Slashed(clients[i], penalty, round, scoresE4[i], thresholdE4);
                emit Blacklisted(clients[i], round);
            } else {
                honest[i]          = true;
                honestTotalScore  += scoresE4[i];
                balances[clients[i]] += stakes[clients[i]];
                stakes[clients[i]]    = 0;
            }
        }
    }

    /**
     * Aggregator submits contribution scores for a round.
     * For each client:
     *   - score >= threshold * equal_share  →  reward proportionally
     *   - score <  threshold * equal_share  →  slash stake, blacklist
     *
     * Slashed tokens are redistributed to honest clients in this round.
     *
     * scoresE4: contribution scores scaled by 10000 (must sum to 10000).
     * rewardPool: tokens to distribute this round (transferred from aggregator).
     */
    // Internal helper: distribute total pool to honest clients.
    function _distribute(
        address[] calldata clients,
        uint256[] calldata scoresE4,
        bool[]    memory   honest,
        uint256            totalPool,
        uint256            honestTotalScore,
        uint256            n,
        uint256            round
    ) internal {
        for (uint i = 0; i < n; i++) {
            if (!honest[i]) continue;
            uint256 share = (honestTotalScore > 0)
                ? (totalPool * scoresE4[i]) / honestTotalScore
                : totalPool / n;
            balances[clients[i]] += share;
            emit Rewarded(clients[i], share, round);
        }
    }

    function submitAndSettle(
        address[] calldata clients,
        uint256[] calldata scoresE4,
        uint256            rewardPool,
        uint256            round
    ) external onlyAggregator {
        require(clients.length == scoresE4.length, "Length mismatch");
        require(balances[aggregator] >= rewardPool, "Insufficient reward pool");

        uint256 n           = clients.length;
        // thresholdE4 = 30% of equal share (3000/n); multiply before divide
        uint256 thresholdE4 = SLASH_THRESHOLD / n;
        bool[]  memory honest = new bool[](n);

        (uint256 slashPool, uint256 honestTotalScore) =
            _classifyAndSlash(clients, scoresE4, thresholdE4, round, honest);

        balances[aggregator] -= rewardPool;
        _distribute(clients, scoresE4, honest,
                    rewardPool + slashPool, honestTotalScore, n, round);
    }

    // ---------------------------------------------------------- view helpers
    function balanceOf(address a) external view returns (uint256) {
        return balances[a];
    }
    function isBlacklisted(address a) external view returns (bool) {
        return blacklisted[a];
    }
    function slashedTotal(address a) external view returns (uint256) {
        return totalSlashed[a];
    }
}
