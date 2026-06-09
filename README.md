# RPCGS

This repository contains an anonymous reference implementation of RPCGS, a reliability-aware personalized cooperative global search method for federated many-task black-box optimization.

The release focuses on the algorithmic core rather than an experiment-management framework. It includes the components needed to inspect and run the RPCGS decision loop:

- behavior-aware client signatures
- personalized server-side routing
- multi-state aggregation for surrogate, generator, and selector states
- heterogeneous candidate-pool construction
- candidate-level transfer reliability estimation
- reliability-aware candidate generation
- feature-based pairwise selection and tournament selection

## Package Layout

- `rpcgs/client.py`: client-side RPCGS loop and candidate selection logic.
- `rpcgs/server.py`: server-side behavior routing and personalized aggregation.
- `rpcgs/utils.py`: reliability, generator, selector, routing, and aggregation utilities.
- `rpcgs/rbf.py`: local radial-basis surrogate model used by RPCGS.
- `rpcgs/framework.py`: minimal in-process interfaces required by the algorithm core.
- `examples/minimal_demo.py`: small synchronous example for a shifted Sphere problem.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Minimal Example

Run the bundled demo from the release root:

```bash
python examples/minimal_demo.py
```

The demo creates several synthetic clients, performs in-process server routing and aggregation, and prints the best objective value found by each client.

## Notes

This package intentionally avoids project-specific launchers, HTTP services, plotting scripts, and experiment-reporting utilities. Those components are not required to understand the RPCGS algorithmic core.
