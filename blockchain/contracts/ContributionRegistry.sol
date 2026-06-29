// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Immutable on-chain registry of client contribution scores per round.
contract ContributionRegistry {

    struct Submission {
        address client;
        uint256 round;
        bytes32 modelHash;      // IPFS CID hash of the model update
        uint256 contributionE4; // contribution score * 10000 (fixed-point)
        uint256 timestamp;
    }

    mapping(uint256 => mapping(address => Submission)) public submissions;
    uint256[] public rounds;
    address public owner;

    event Submitted(address indexed client, uint256 indexed round,
                    bytes32 modelHash, uint256 contributionE4);

    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }

    constructor() { owner = msg.sender; }

    function submit(uint256 round, bytes32 modelHash, uint256 contributionE4)
        external
    {
        submissions[round][msg.sender] = Submission(
            msg.sender, round, modelHash, contributionE4, block.timestamp);
        emit Submitted(msg.sender, round, modelHash, contributionE4);
    }

    function getSubmission(uint256 round, address client)
        external view returns (Submission memory)
    {
        return submissions[round][client];
    }
}
