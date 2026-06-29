"""
Deploy all three contracts to Ganache local testnet.
Run: python blockchain/deploy.py
Outputs deployed_contracts.json with addresses.
"""

import json
from web3 import Web3
from solcx import compile_source, install_solc

GANACHE_URL = "http://127.0.0.1:8545"
INITIAL_TOKEN_SUPPLY = 1_000_000


def compile_contract(source_path):
    install_solc("0.8.0")
    with open(source_path) as f:
        source = f.read()
    compiled = compile_source(source, output_values=["abi", "bin"],
                               solc_version="0.8.0")
    key = list(compiled.keys())[0]
    return compiled[key]["abi"], compiled[key]["bin"]


def deploy(w3, abi, bytecode, *constructor_args):
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = contract.constructor(*constructor_args).transact(
        {"from": w3.eth.accounts[0], "gas": 3_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.contractAddress


def main():
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    assert w3.is_connected(), "Start Ganache first: npx ganache"

    contracts = {}
    for name, args in [
        ("ContributionRegistry", []),
        ("ConsensusContract", []),
        ("RewardDistributor", [INITIAL_TOKEN_SUPPLY]),
    ]:
        path = f"blockchain/contracts/{name}.sol"
        abi, bytecode = compile_contract(path)
        addr = deploy(w3, abi, bytecode, *args)
        contracts[name] = {"address": addr, "abi": abi}
        print(f"Deployed {name} at {addr}")

    with open("deployed_contracts.json", "w") as f:
        json.dump(contracts, f, indent=2)
    print("Saved to deployed_contracts.json")


if __name__ == "__main__":
    main()
