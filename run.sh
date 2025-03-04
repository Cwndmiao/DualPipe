#!/bin/bash
set -ex

nsys profile -t cuda,nvtx -o dualpipe --force-overwrite=true \
    env MASTER_ADDR=localhost MASTER_PORT=6789 python example.py
