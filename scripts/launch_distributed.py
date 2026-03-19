#!/usr/bin/env python3
"""Distributed training launcher for WorldFlux.

Convenience wrapper around torchrun for multi-GPU DDP training.

Usage:
    # Direct launch with torchrun (recommended):
    torchrun --nproc_per_node=4 -m worldflux train --use-ddp

    # Using this launcher script:
    python scripts/launch_distributed.py --nproc-per-node 4 --config worldflux.toml

    # With additional training arguments:
    python scripts/launch_distributed.py --nproc-per-node 2 \\
        --config worldflux.toml --steps 50000

Note:
    This script requires ``torch.distributed.run`` (torchrun) to be available.
    It sets up the necessary environment variables and delegates to torchrun.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch distributed WorldFlux training via torchrun.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of processes per node (typically number of GPUs).",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes for multi-node training.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Rank of this node in multi-node training.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="Master node address.",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="29500",
        help="Master node port.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="worldflux.toml",
        help="Path to worldflux.toml configuration file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu).",
    )

    args, extra_args = parser.parse_known_args()

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "-m",
        "worldflux",
        "train",
        "--config",
        args.config,
        "--use-ddp",
    ]

    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    if args.device is not None:
        cmd.extend(["--device", args.device])

    cmd.extend(extra_args)

    print(f"Launching distributed training: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
