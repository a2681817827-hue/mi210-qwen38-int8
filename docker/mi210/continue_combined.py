"""Continue the serial local campaign, prioritizing NS17 fast decode."""
import json
from pathlib import Path
import shutil
import subprocess
import time


root = Path(__file__).resolve().parents[2]
base = root / 'logs/mi210/continue-20261005'
report = base / 'head-search/campaign.json'
while json.loads(report.read_text())['status'] == 'RUNNING':
    time.sleep(5)
if json.loads(report.read_text())['status'] != 'MEASUREMENTS_COMPLETE_SELECTION_PENDING':
    raise RuntimeError('Inspect the previous campaign before allocating the GPU')


def gate(name, image, filename, arguments):
    command = ['docker', 'run', '--name', name, '--restart=no', '--device=/dev/kfd',
               '--device=/dev/dri/renderD128', '--ipc=host', '-e', 'HIP_VISIBLE_DEVICES=0',
               '-v', str(root / 'docker/mi210') + ':/tmp/tools:ro',
               '-v', str(base) + ':/results', '--entrypoint',
               '/opt/int8-vllm/.venv/bin/python', image, '/tmp/tools/' + filename, *arguments]
    with (base / (name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def campaign(directory, image, arms, source):
    for arm in arms:
        destination = Path('/data/cache/mi210-vllm') / ('mi210-' + directory + '-' + arm)
        if not destination.exists():
            shutil.copytree(source, destination, symlinks=True)
    subprocess.run([str(root / '.venv/bin/python'), str(root / 'docker/mi210/serving_campaign.py'),
                    '--output-dir', str(base / directory), '--image', image,
                    '--arms', *arms, '--reference-quality',
                    str(base / 'serving/original/quality.json'), '--repetitions', '3'], check=True)


gate('mi210-ns17-flash-gate', 'int8-vllm-mi210:ck-candidates-20261005',
     'graph_split_gate.py', ['--max-query', '18', '--splits', '16',
                             '--output', '/results/ns17-flash-gate.jsonl'])
campaign('flash17', 'int8-vllm-mi210:ck-candidates-20261005', ['head-ns17-flash'],
         '/data/cache/mi210-vllm/mi210-head-search-head-ns17')
gate('mi210-gdn-tiny-ba-gate', 'int8-vllm-mi210:gdn-candidates-20261005',
     'gate_tiny_ba.py', ['--output', '/results/gdn-tiny-ba-gate.json'])
campaign('gdn-search', 'int8-vllm-mi210:gdn-candidates-20261005',
         ['head-tiny-ba', 'head-zview', 'head-split32'],
         '/data/cache/mi210-vllm/mi210-serving-int8-head')
