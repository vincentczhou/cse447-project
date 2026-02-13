#!/usr/bin/env bash
set -e
set -v
uv run --no-dev python src/myprogram.py test --work_dir work --test_data $1 --test_output $2
