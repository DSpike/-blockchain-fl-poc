// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Distributes ERC20-style token rewards proportional to contribution scores.
contract RewardDistributor {

    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    address public owner;

    event RewardDistributed(address indexed client, uint256 amount, uint256 round);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }

    constructor(uint256 initialSupply) {
        owner = msg.sender;
        totalSupply = initialSupply;
        balances[owner] = initialSupply;
    }

    /// Distribute rewardPool tokens across clients weighted by contributionE4 (fixed-point *10000).
    function distribute(
        address[] calldata clients,
        uint256[] calldata contributionsE4,
        uint256 rewardPool,
        uint256 round
    ) external onlyOwner {
        require(clients.length == contributionsE4.length, "Length mismatch");
        require(balances[owner] >= rewardPool, "Insufficient pool");

        uint256 totalContrib = 0;
        for (uint i = 0; i < contributionsE4.length; i++) totalContrib += contributionsE4[i];
        require(totalContrib > 0, "Zero contributions");

        balances[owner] -= rewardPool;
        for (uint i = 0; i < clients.length; i++) {
            uint256 reward = (rewardPool * contributionsE4[i]) / totalContrib;
            balances[clients[i]] += reward;
            emit RewardDistributed(clients[i], reward, round);
        }
    }

    function balanceOf(address account) external view returns (uint256) {
        return balances[account];
    }
}
