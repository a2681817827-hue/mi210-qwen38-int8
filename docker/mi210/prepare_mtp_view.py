"""Create a symlinked checkpoint view for explicitly unquantized MTP weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def prepare(source, output):
    source = source.resolve()
    if output.exists() or output.parent.resolve() != source.parent:
        raise ValueError('Use a fresh sibling directory under the mounted model root')
    config = json.loads((source / 'config.json').read_text())
    weight_map = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    names = [name for name in weight_map if name.startswith('mtp.')]
    if not names or not config.get('text_config', config).get('mtp_num_hidden_layers'):
        raise ValueError('Checkpoint must contain configured MTP layers and weights')
    tensors = {}
    for shard in sorted({weight_map[name] for name in names}):
        with (source / shard).open('rb') as file:
            header = json.loads(file.read(struct.unpack('<Q', file.read(8))[0]))
        for name in names:
            if weight_map[name] == shard:
                metadata = header[name]
                if not name.endswith('.weight') or metadata['dtype'] not in ('BF16', 'F16', 'F32'):
                    raise ValueError('This view supports only unquantized MTP weights')
                tensors[name] = {k: metadata[k] for k in ('dtype', 'shape')}
    quant = config['quantization_config']
    if quant.get('quant_method') != 'gptq' or quant.get('bits') != 8:
        raise ValueError('Expected the local INT8 GPTQ target checkpoint')
    quant['dynamic'] = {'-:.*mtp.*': {}, **quant.get('dynamic', {})}
    output.mkdir()
    for path in source.iterdir():
        if path.name != 'config.json':
            (output / path.name).symlink_to(os.path.relpath(path, output))
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest = {'source': str(source), 'view': str(output), 'weights_copied': False,
                'source_config_sha256': hashlib.sha256((source / 'config.json').read_bytes()).hexdigest(),
                'view_config_sha256': hashlib.sha256((output / 'config.json').read_bytes()).hexdigest(),
                'mtp_tensors': tensors,
                'note': 'Only MTP quantization metadata changes; target weights and source config stay intact.'}
    (output / 'mi210-mtp-view.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
