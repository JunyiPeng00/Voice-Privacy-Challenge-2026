#!/bin/bash
# Download the heads of the text attacker (see text_attacker/README.md) to exp/text_attacker/heads.

set -e

repo=https://huggingface.co/JYP2024/vpc2026-text-attacker/resolve/b1ade524323405b1595508a4536adc5af956267b
heads_dir=exp/text_attacker/heads

for head in libri en de es fr; do
    if [ ! -f $heads_dir/$head/head.safetensors ] || [ ! -f $heads_dir/$head/head.json ]; then
        echo "Download text attacker head '$head'..."
        mkdir -p $heads_dir/$head
        for file in head.safetensors head.json; do
            # HF_TOKEN is only needed while the model repository is not public
            wget -nv ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} -O $heads_dir/$head/$file $repo/$head/$file
        done
    fi
done

