# Evaluation of the text attacker: speaker verification from transcripts only.
# Usage and the meaning of the numbers are described in text_attacker/README.md.
# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
import os
import json
import time
from argparse import ArgumentParser
from importlib import metadata
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--config', default='configs/track1/eval_text.yaml')
parser.add_argument('--overwrite', type=str, default='{}')
parser.add_argument('--force_compute', type=str, default='False')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
else: # CUDA_VISIBLE_DEVICES more important than the gpu_ids arg
    args.gpu_ids = ",".join([ str(i) for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(","))])

import torch

from utils import parse_yaml, setup_logger
from text_attacker.official import evaluate_text_attacker

logger = setup_logger(__name__)


def package_versions():
    versions = {}
    for package in ('torch', 'transformers', 'tokenizers', 'speechbrain', 'safetensors', 'numpy'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


if __name__ == '__main__':
    params = parse_yaml(Path(args.config), overrides=json.loads(args.overwrite))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert params['data_dir'].exists(), f'{params["data_dir"]} does not exist'

    logger.info('================================')
    logger.info('Perform text attacker evaluation')
    logger.info('================================')
    start_time = time.time()
    evaluate_text_attacker(params=params, device=device, force_compute=args.force_compute.lower() == 'true')
    logger.info("--- Text attacker evaluation time: %f min ---" % (float(time.time() - start_time) / 60))

    results_dir = params['privacy']['text_attacker']['evaluation']['results_dir']
    anon_suffix = params.get('anon_data_suffix') or ''
    with open(results_dir / f'run_info{anon_suffix}.json', 'w') as f:
        json.dump({'config': str(args.config), 'overwrite': json.loads(args.overwrite), 'device': device,
                   'backbone': str(params['privacy']['text_attacker']['backbone']),
                   'backbone_revision': params['privacy']['text_attacker'].get('backbone_revision'),
                   'versions': package_versions()}, f, indent=2)
