# RPCGS

This repository contains an anonymous reference implementation of RPCGS, a method for federated many-task expensive black-box optimization. Each client owns one expensive task; clients never share data or surrogate predictions, only proposals of where good solutions may lie.

The algorithmic core includes:

- proposal-level transfer: transferred information only adds candidate points; every candidate is scored by expected improvement under the client's own surrogate
- a consensus proposal network: each client fits a conditional generator on its elite points, and the server averages the generators of all clients
- individual elite centers: each client shares the mean and spread of its best points, and every client proposes points around the centers of all clients
- a local least-squares RBF surrogate with bootstrap uncertainty, an adaptive local region around the incumbent, and boundary candidates

Because the uniform, local-region and boundary candidates do not depend on other clients and the selection uses the local model only, transferred proposals cannot lower the local expected improvement of the chosen point.

## Package Layout

- `rpcgs/client.py`: client loop, candidate pool (uniform, local region, transfer proposals, boundary) and EI selection.
- `rpcgs/server.py`: averaging of generator states and forwarding of elite centers.
- `rpcgs/utils.py`: least-squares RBF network, conditional generator and helpers.
- `rpcgs/framework.py`: minimal in-process client, server and versioned buffer interfaces.
- `examples/minimal_demo.py`: an in-process example with shifted Sphere tasks.

## Installation

Use Python 3.10 or newer and install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is optional; without it the generator falls back to a Gaussian around the weighted elite mean.

## Minimal Example

Run the demo from the repository root:

```bash
python examples/minimal_demo.py
```

The demo runs four clients in threads against an in-process server and prints each client's evaluation count and best objective value. Per-round diagnostics (candidate source, proposal arm, EI) are written to a temporary directory.

## Main Parameters

Client: `candidate_random` 96, `candidate_perturb` 48, `candidate_generate` 64, `candidate_boundary` 32, `proposal` hybrid, `hybrid_gen_share` 0.75, `center_alloc` uniform, `bootstrap_num` 5, `y_norm` zscore.
Server: `collaborate` true, `d_share_size` 50.
The defaults in the class docstrings define the method; the remaining switches are ablations.

## Notes

This standalone release focuses on the algorithmic core. The demo illustrates the decision loop; full paper experiments are maintained separately.
