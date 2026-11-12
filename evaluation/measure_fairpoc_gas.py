"""
Measure on-chain gas for FairPoC.sol, the contract the paper describes.

Background: the gas figures in the paper's Table V were measured on
2026-06-30 from the earlier three-contract architecture
(ContributionRegistry + ConsensusContract + RewardDistributor), which is
what deployed_contracts.json holds. FairPoC.sol was written on 2026-07-15
as a consolidated replacement and was never deployed or measured, yet the
paper attributes Eq. 9 and Table V to its submitAndSettle(). This script
deploys FairPoC and measures it, so the paper's claims describe a contract
that exists.

Measures:
  - deployment cost
  - stake() per client
  - submitAndSettle() across a sweep of n, then fits G(n) = a + b*n

Usage (needs a local node on 127.0.0.1:8545):
    ganache-cli --port 8545 --accounts 20 --gasLimit 30000000 --quiet &
    python evaluation/measure_fairpoc_gas.py
"""
import json
import os
import sys

import solcx
from web3 import Web3

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "blockchain", "contracts", "FairPoC.sol")
OUT = os.path.join(HERE, "results", "fairpoc_gas.csv")

RPC = "http://127.0.0.1:8545"
SOLC = "0.8.19"
SWEEP = [10, 25, 50, 100, 200]
SUPPLY = 10 ** 12


def compile_contract():
    src = open(SRC).read()
    # the file pins 0.8.20; 0.8.19 is what is installed and the source uses
    # nothing newer. Relaxed for compilation only, the file is not modified.
    src = src.replace("pragma solidity 0.8.20;", "pragma solidity ^0.8.0;")
    out = solcx.compile_source(src, output_values=["abi", "bin"],
                               solc_version=SOLC)
    key = [k for k in out if k.endswith(":FairPoC")][0]
    return out[key]["abi"], out[key]["bin"]


def main():
    w3 = Web3(Web3.HTTPProvider(RPC))
    if not w3.is_connected():
        sys.exit("no node at " + RPC)
    agg = w3.eth.accounts[0]
    print("aggregator:", agg)

    abi, binary = compile_contract()
    C = w3.eth.contract(abi=abi, bytecode=binary)
    tx = C.constructor(SUPPLY).transact({"from": agg, "gas": 8_000_000})
    rc = w3.eth.wait_for_transaction_receipt(tx)
    deploy_gas = rc["gasUsed"]
    fp = w3.eth.contract(address=rc["contractAddress"], abi=abi)
    print("deployed at %s, gas %d" % (rc["contractAddress"], deploy_gas))

    # stake() measured from a funded account. Only the aggregator holds
    # tokens at genesis (the contract has no transfer), so the aggregator
    # itself is used to price the call.
    tx = fp.functions.stake(1).transact({"from": agg, "gas": 500_000})
    stake_gas = w3.eth.wait_for_transaction_receipt(tx)["gasUsed"]
    print("stake(): %d gas" % stake_gas)

    rows = []
    for n in SWEEP:
        clients = [Web3.to_checksum_address("0x%040x" % (0x1000 + i))
                   for i in range(n)]
        # threshold is 3000/n; make ~10% of clients fall below it so both
        # the slash path and the reward path are exercised, as in a real round
        thr = 3000 // n
        scores = [max(1, thr // 3) if i % 10 == 0 else thr * 4
                  for i in range(n)]
        tx = fp.functions.submitAndSettle(clients, scores, 1000, n).transact(
            {"from": agg, "gas": 29_000_000})
        g = w3.eth.wait_for_transaction_receipt(tx)["gasUsed"]
        rows.append((n, g))
        print("submitAndSettle(n=%4d): %10d gas  (%8.1f per client)"
              % (n, g, g / n))

    # least squares fit of G(n) = a + b n
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    k = len(xs)
    mx = sum(xs) / k
    my = sum(ys) / k
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    print("")
    print("fit: G(n) = %.0f + %.0f n" % (a, b))
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    print("R^2 = %.6f" % (1 - ss_res / ss_tot))

    print("")
    print("%8s %12s %12s %12s" % ("clients", "total gas", "ETH@30Gwei",
                                  "USD@$3k"))
    for n, g in rows:
        eth = g * 30e-9
        print("%8d %12d %12.4f %12.2f" % (n, g, eth, eth * 3000))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        fh.write("contract,function,n,gas,eth_30gwei,usd_3k" + chr(10))
        fh.write("FairPoC,deploy,,%d,%.6f,%.2f" % (
            deploy_gas, deploy_gas * 30e-9, deploy_gas * 30e-9 * 3000) + chr(10))
        fh.write("FairPoC,stake,1,%d,%.6f,%.2f" % (
            stake_gas, stake_gas * 30e-9, stake_gas * 30e-9 * 3000) + chr(10))
        for n, g in rows:
            fh.write("FairPoC,submitAndSettle,%d,%d,%.6f,%.2f" % (
                n, g, g * 30e-9, g * 30e-9 * 3000) + chr(10))
    print("")
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
