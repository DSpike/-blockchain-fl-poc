// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Miner voting contract: miners propose aggregated model hashes and vote.
/// Consensus requires >= 67% weighted vote share.
contract ConsensusContract {

    struct Proposal {
        address miner;
        bytes32 globalModelHash;
        uint256 voteWeight;   // accumulated weighted votes
        bool finalised;
    }

    uint256 public constant CONSENSUS_THRESHOLD_PCT = 67;
    uint256 public currentRound;
    address public owner;

    mapping(uint256 => Proposal[]) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    mapping(address => uint256) public minerStake;
    mapping(address => uint256) public minerReputation; // 0-100

    bytes32 public latestGlobalModel;

    event ProposalSubmitted(uint256 indexed round, address miner, bytes32 modelHash);
    event ConsensusReached(uint256 indexed round, bytes32 globalModelHash);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }

    constructor() { owner = msg.sender; currentRound = 0; }

    function registerMiner(address miner, uint256 stake) external onlyOwner {
        minerStake[miner] = stake;
        minerReputation[miner] = 50; // start at neutral reputation
    }

    function submitProposal(uint256 round, bytes32 modelHash) external {
        require(minerStake[msg.sender] > 0, "Not a registered miner");
        proposals[round].push(Proposal(msg.sender, modelHash, 0, false));
        emit ProposalSubmitted(round, msg.sender, modelHash);
    }

    function vote(uint256 round, uint256 proposalIndex) external {
        require(!hasVoted[round][msg.sender], "Already voted");
        require(minerStake[msg.sender] > 0, "Not a registered miner");
        hasVoted[round][msg.sender] = true;

        uint256 weight = minerStake[msg.sender] * minerReputation[msg.sender];
        proposals[round][proposalIndex].voteWeight += weight;

        _checkConsensus(round);
    }

    function _checkConsensus(uint256 round) internal {
        uint256 totalWeight = 0;
        for (uint i = 0; i < proposals[round].length; i++) {
            totalWeight += proposals[round][i].voteWeight;
        }
        for (uint i = 0; i < proposals[round].length; i++) {
            Proposal storage p = proposals[round][i];
            if (!p.finalised && totalWeight > 0 &&
                p.voteWeight * 100 / totalWeight >= CONSENSUS_THRESHOLD_PCT) {
                p.finalised = true;
                latestGlobalModel = p.globalModelHash;
                emit ConsensusReached(round, p.globalModelHash);
            }
        }
    }
}
