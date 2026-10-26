"""Python <-> Ethereum smart contract interface via Web3.py."""

from web3 import Web3
import json, os


def connect(ganache_url="http://127.0.0.1:8545"):
    w3 = Web3(Web3.HTTPProvider(ganache_url))
    assert w3.is_connected(), f"Cannot connect to Ganache at {ganache_url}"
    return w3


def load_contract(w3, abi_path, address):
    with open(abi_path) as f:
        abi = json.load(f)
    return w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)


def submit_contribution(contract, client_addr, round_num, model_hash_hex, contribution_score, owner_addr, gas=300_000):
    """Store a client's contribution proof on-chain."""
    contrib_e4 = int(contribution_score * 10_000)
    clean_hex = model_hash_hex[2:] if model_hash_hex.startswith("0x") else model_hash_hex
    model_hash = bytes.fromhex(clean_hex)[:32]
    tx = contract.functions.submit(round_num, model_hash, contrib_e4).transact(
        {"from": client_addr, "gas": gas})
    return tx


def distribute_rewards(contract, clients, scores, reward_pool, round_num, owner_addr, gas=500_000):
    """Trigger on-chain reward distribution."""
    contribs_e4 = [int(s * 10_000) for s in scores]
    tx = contract.functions.distribute(
        clients, contribs_e4, reward_pool, round_num
    ).transact({"from": owner_addr, "gas": gas})
    return tx
