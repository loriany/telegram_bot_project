#!/usr/bin/env bash
set -e

python download_models.py
python telegram_bot_inference.py